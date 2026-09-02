"""Analysis for the self/other boundary study. Runs on data/rollouts.jsonl, emits results.md
and plots under data/plots/.

Written and tested against synthetic data (see `_make_synthetic_rollouts` / `--self-test`)
BEFORE the main study was run, per the spec's pre-registration requirement: this script's
logic was fixed before any real main-study data existed.

Usage:
    uv run python analyze.py                 # analyze data/rollouts.jsonl -> results.md
    uv run python analyze.py --self-test      # run on synthetic data, print sanity checks
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CELL_NAMES, N_ROUNDS, SUBJECT_MODEL, FOREIGN_MODEL

DATA_DIR = Path(__file__).parent / "data"
ROLLOUTS_PATH = DATA_DIR / "rollouts.jsonl"
PLOTS_DIR = DATA_DIR / "plots"
RESULTS_PATH = Path(__file__).parent / "results.md"
SPEND_PATH = Path(__file__).parent / "spend.json"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

IDENTITY_RE = re.compile(
    r"\b(model|ai|gpt|instance|clone|myself|itself|identical|same model|different model)\b",
    re.IGNORECASE,
)
TEST_AWARENESS_RE = re.compile(
    r"\b(experiment|study|research(er)?|test(ing|ed)?|being tested|evaluat\w*)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rollouts(path: Path = ROLLOUTS_PATH) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def by_phase(records: list[dict], phase: str) -> list[dict]:
    return [r for r in records if r["phase"] == phase]


def ok_only(records: list[dict]) -> list[dict]:
    return [r for r in records if r["status"] == "ok"]


# ---------------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------------
def bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def cell_means_and_ci(records: list[dict], cell_names: list[str] = CELL_NAMES) -> dict:
    out = {}
    for cell in cell_names:
        cell_records = [r for r in records if r["cell"] == cell]
        ok = ok_only(cell_records)
        n_invalid = len(cell_records) - len(ok)
        values = [r["mean_contribution"] for r in ok]
        mean, lo, hi = bootstrap_ci(values)
        out[cell] = {
            "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "n": len(cell_records), "n_ok": len(ok), "n_invalid": n_invalid,
            "values": values,
        }
    return out


def contrasts_bootstrap(records: list[dict], n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstraps within each of the four 2x2 cells (resample rollouts with replacement inside
    each cell), recomputes the three contrasts each iteration, and reports the point estimate
    (on the real data) plus a percentile 95% CI from the resample distribution."""
    cell_values = {
        cell: np.array([r["mean_contribution"] for r in ok_only(records) if r["cell"] == cell])
        for cell in ["self_self", "self_other", "other_self", "other_other"]
    }
    for cell, arr in cell_values.items():
        if len(arr) == 0:
            raise ValueError(f"No valid rollouts for cell {cell!r}; cannot compute contrasts.")

    def contrasts_from_means(m):
        ss, so, os_, oo = m["self_self"], m["self_other"], m["other_self"], m["other_other"]
        label_effect = (ss + os_) - (so + oo)
        true_source_effect = (ss + so) - (os_ + oo)
        interaction = (ss - so) - (os_ - oo)
        return label_effect, true_source_effect, interaction

    point_means = {cell: arr.mean() for cell, arr in cell_values.items()}
    point_label, point_source, point_interaction = contrasts_from_means(point_means)

    rng = np.random.default_rng(seed)
    boot_label = np.empty(n_resamples)
    boot_source = np.empty(n_resamples)
    boot_interaction = np.empty(n_resamples)
    resampled_means = {}
    for cell, arr in cell_values.items():
        samples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
        resampled_means[cell] = samples.mean(axis=1)
    for i in range(n_resamples):
        m = {cell: resampled_means[cell][i] for cell in cell_values}
        boot_label[i], boot_source[i], boot_interaction[i] = contrasts_from_means(m)

    def summarize(point, boot):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "label_effect": summarize(point_label, boot_label),
        "true_source_effect": summarize(point_source, boot_source),
        "interaction": summarize(point_interaction, boot_interaction),
    }


def noise_floor_stats(records: list[dict]) -> dict:
    nf = ok_only(by_phase(records, "noise_floor"))
    variant_values: dict[int, list[float]] = {}
    for r in nf:
        variant_values.setdefault(r["variant_idx"], []).append(r["mean_contribution"])
    variant_means = {v: float(np.mean(vals)) for v, vals in variant_values.items()}
    if not variant_means:
        return {"variant_means": {}, "range": float("nan"), "sd": float("nan"), "n_variants": 0}
    means_arr = np.array(list(variant_means.values()))
    return {
        "variant_means": variant_means,
        "range": float(means_arr.max() - means_arr.min()),
        "sd": float(means_arr.std(ddof=1)) if len(means_arr) > 1 else float("nan"),
        "n_variants": len(variant_means),
        "n_rollouts": len(nf),
    }


def trajectories(records: list[dict], cell_names: list[str] = CELL_NAMES) -> dict:
    out = {}
    for cell in cell_names:
        ok = ok_only([r for r in records if r["cell"] == cell])
        per_round = {rd: [] for rd in range(1, N_ROUNDS + 1)}
        for r in ok:
            for rd_data in r["rounds"]:
                per_round[rd_data["round"]].append(rd_data["subject_c"])
        out[cell] = [float(np.mean(per_round[rd])) if per_round[rd] else float("nan")
                     for rd in range(1, N_ROUNDS + 1)]
    return out


def secondary_covariates(records: list[dict], cell_names: list[str] = CELL_NAMES) -> dict:
    out = {}
    for cell in cell_names:
        ok = ok_only([r for r in records if r["cell"] == cell])
        n = len(ok)
        identity_flags = []
        test_flags = []
        for r in ok:
            reasons = " ".join(rd.get("reason") or "" for rd in r["rounds"])
            identity_flags.append(bool(IDENTITY_RE.search(reasons)))
            test_flags.append(bool(TEST_AWARENESS_RE.search(reasons)))
        out[cell] = {
            "n": n,
            "identity_mention_frac": (sum(identity_flags) / n) if n else float("nan"),
            "test_awareness_frac": (sum(test_flags) / n) if n else float("nan"),
        }
    return out


def invalid_rate(records: list[dict]) -> dict:
    n_total = len(records)
    n_invalid = len([r for r in records if r["status"] == "invalid"])
    return {"n_total": n_total, "n_invalid": n_invalid,
            "rate": (n_invalid / n_total) if n_total else float("nan")}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(cell_stats: dict, traj: dict, nf: dict, out_dir: Path = PLOTS_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Cell means with bootstrap CI
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cells = list(cell_stats.keys())
    means = [cell_stats[c]["mean"] for c in cells]
    lo = [cell_stats[c]["mean"] - cell_stats[c]["ci_lo"] for c in cells]
    hi = [cell_stats[c]["ci_hi"] - cell_stats[c]["mean"] for c in cells]
    ax.bar(cells, means, yerr=[lo, hi], capsize=4, color="#4C72B0")
    ax.set_ylabel("Mean contribution (0-10)")
    ax.set_title("Cell means with bootstrap 95% CI")
    ax.set_ylim(0, 10)
    plt.xticks(rotation=20)
    fig.tight_layout()
    p = out_dir / "cell_means.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # Trajectories
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cell, vals in traj.items():
        ax.plot(range(1, N_ROUNDS + 1), vals, marker="o", label=cell)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean contribution")
    ax.set_title("Per-round trajectories by cell")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "trajectories.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # Noise floor
    if nf["variant_means"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vidx = sorted(nf["variant_means"].keys())
        vvals = [nf["variant_means"][v] for v in vidx]
        ax.bar([str(v) for v in vidx], vvals, color="#888888")
        ax.axhline(np.mean(vvals), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Noise-floor prompt variant")
        ax.set_ylabel("Mean contribution")
        ax.set_title(f"Noise floor across {nf['n_variants']} paraphrase variants "
                     f"(range={nf['range']:.2f}, sd={nf['sd']:.2f})")
        ax.set_ylim(0, 10)
        fig.tight_layout()
        p = out_dir / "noise_floor.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))

    return paths


# ---------------------------------------------------------------------------
# Results.md
# ---------------------------------------------------------------------------
def write_results_md(cell_stats, contrasts, nf, traj, covariates, invalid, spend, path=RESULTS_PATH):
    lines = []
    lines.append("# Results: self/other boundary study")
    lines.append("")
    lines.append(f"Subject model: `{SUBJECT_MODEL}`. Foreign model: `{FOREIGN_MODEL}`.")
    lines.append("")
    lines.append(f"**Actual total OpenAI spend: ${spend:.4f}** (hard budget $7.57, kill switch $6.00).")
    lines.append("")
    lines.append(f"**Invalid-parse rate (main study): {invalid['n_invalid']}/{invalid['n_total']} "
                 f"= {invalid['rate']:.3%}** (rollouts abandoned after a JSON parse retry failed).")
    lines.append("")

    lines.append("## Noise floor (spec section 5)")
    lines.append("")
    if nf["variant_means"]:
        lines.append(f"20 semantically-equivalent rewordings of the baseline prompt, "
                     f"{nf.get('n_rollouts', 0)} rollouts. Spread of per-variant mean "
                     f"contribution: **range = {nf['range']:.3f}, sd = {nf['sd']:.3f}** "
                     f"(0-10 scale). Any between-cell effect below this spread should be read "
                     f"as prompt sensitivity, not a self/other effect.")
    else:
        lines.append("No noise-floor data available.")
    lines.append("")
    lines.append("![noise floor](data/plots/noise_floor.png)")
    lines.append("")

    lines.append("## Cell means (bootstrap 95% CI, 10,000 resamples)")
    lines.append("")
    lines.append("| Cell | Mean | 95% CI | n (ok/invalid) |")
    lines.append("|---|---|---|---|")
    for cell, s in cell_stats.items():
        lines.append(f"| `{cell}` | {s['mean']:.3f} | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | "
                     f"{s['n_ok']}/{s['n_invalid']} |")
    lines.append("")
    lines.append("![cell means](data/plots/cell_means.png)")
    lines.append("")

    lines.append("## Contrasts, vs. the noise floor")
    lines.append("")
    nf_range = nf["range"] if nf["variant_means"] else float("nan")
    nf_sd = nf["sd"] if nf["variant_means"] else float("nan")
    lines.append("| Contrast | Point estimate | 95% CI | Noise floor range | Noise floor sd | "
                 "Above noise floor? |")
    lines.append("|---|---|---|---|---|---|")
    for name, key in [("Label main effect", "label_effect"),
                       ("True-source main effect", "true_source_effect"),
                       ("Interaction", "interaction")]:
        c = contrasts[key]
        above = abs(c["point"]) > nf_range if nf_range == nf_range else float("nan")
        lines.append(f"| {name} | {c['point']:.3f} | [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] | "
                     f"{nf_range:.3f} | {nf_sd:.3f} | {'yes' if above else 'no'} |")
    lines.append("")
    lines.append("Label main effect = (self_self + other_self) - (self_other + other_other). "
                 "True-source main effect = (self_self + self_other) - (other_self + other_other). "
                 "Interaction = (self_self - self_other) - (other_self - other_other).")
    lines.append("")

    lines.append("## Per-round trajectories")
    lines.append("")
    lines.append("![trajectories](data/plots/trajectories.png)")
    lines.append("")

    lines.append("## Secondary covariates (from `reason` text, regex/keyword, no extra API calls)")
    lines.append("")
    lines.append("| Cell | n | Fraction mentioning partner identity | Fraction showing test-awareness |")
    lines.append("|---|---|---|---|")
    for cell, cov in covariates.items():
        lines.append(f"| `{cell}` | {cov['n']} | {cov['identity_mention_frac']:.3f} | "
                     f"{cov['test_awareness_frac']:.3f} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------
def _make_synthetic_rollouts(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    cell_base = {
        "baseline": 5.0, "self_self": 5.8, "self_other": 4.9,
        "other_self": 5.1, "other_other": 4.3,
    }
    for cell, base in cell_base.items():
        for i in range(80):
            rounds = []
            for rd in range(1, N_ROUNDS + 1):
                c = int(np.clip(round(base + rng.normal(0, 1.5)), 0, 10))
                rounds.append({"round": rd, "subject_c": c, "partner_c": 5,
                               "reason": "cooperating steadily"})
            records.append({
                "rollout_id": f"synthetic:{cell}:{i}", "phase": "main", "cell": cell,
                "variant_idx": None, "status": "ok", "model": "synthetic",
                "rounds": rounds,
                "mean_contribution": float(np.mean([r["subject_c"] for r in rounds])),
            })
    for v in range(20):
        variant_base = 5.0 + rng.normal(0, 0.3)
        for i in range(8):
            rounds = [{"round": rd, "subject_c": int(np.clip(round(variant_base + rng.normal(0, 1.5)), 0, 10)),
                       "partner_c": 5, "reason": "ok"} for rd in range(1, N_ROUNDS + 1)]
            records.append({
                "rollout_id": f"synthetic_nf:{v}:{i}", "phase": "noise_floor", "cell": "baseline",
                "variant_idx": v, "status": "ok", "model": "synthetic", "rounds": rounds,
                "mean_contribution": float(np.mean([r["subject_c"] for r in rounds])),
            })
    return records


def self_test():
    records = _make_synthetic_rollouts()
    main_records = by_phase(records, "main")
    cell_stats = cell_means_and_ci(main_records)
    contrasts = contrasts_bootstrap(main_records)
    nf = noise_floor_stats(records)
    traj = trajectories(main_records)
    covariates = secondary_covariates(main_records)
    invalid = invalid_rate(main_records)
    print("cell_stats:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "values"}
                                      for k, v in cell_stats.items()}, indent=2))
    print("contrasts:", json.dumps(contrasts, indent=2))
    print("noise_floor:", {k: v for k, v in nf.items() if k != "variant_means"})
    print("invalid:", invalid)
    assert cell_stats["self_self"]["mean"] > cell_stats["other_other"]["mean"], \
        "sanity check failed: synthetic self_self should exceed other_other"
    print("\nSELF-TEST OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    records = load_rollouts()
    main_records = by_phase(records, "main")
    if not main_records:
        print("No main-study rollouts found in data/rollouts.jsonl. Run `run.py --main` first.")
        return

    cell_stats = cell_means_and_ci(main_records)
    contrasts = contrasts_bootstrap(main_records)
    nf = noise_floor_stats(records)
    traj = trajectories(main_records)
    covariates = secondary_covariates(main_records)
    invalid = invalid_rate(main_records)
    spend = json.loads(SPEND_PATH.read_text(encoding="utf-8"))["total_cost_usd"] if SPEND_PATH.exists() else float("nan")

    make_plots(cell_stats, traj, nf)
    out_path = write_results_md(cell_stats, contrasts, nf, traj, covariates, invalid, spend)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
