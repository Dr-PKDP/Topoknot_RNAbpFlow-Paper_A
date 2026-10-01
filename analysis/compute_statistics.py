"""
compute_statistics.py -- target-level and pooled-dimension statistical
tests for the primary cohort-wide finding (manuscript Methods S2.5,
Results S3.3-S3.4).

Reads raw per-dimension signal/floor values directly from
results/stage0_analysis_workbook.xlsx (sheet Finding1_RawData) rather
than a hardcoded copy of the numbers, so this script and that workbook
cannot drift apart.

Computes:
  1. Target-level Delta = angle-clearing proportion - distance-clearing
     proportion, per target; mean and median across targets; 95%
     bootstrap confidence interval on the mean (target-level resampling
     with replacement, 10,000 resamples, percentile method); exact sign
     test on targets with non-zero Delta.
  2. Pooled-dimension analysis: every coordinate dimension across the
     cohort as one matched distance/angle pair; McNemar's exact test;
     Wilson 95% confidence intervals on the two pooled clearing
     proportions.
  3. Leave-one-target-out sensitivity: pooled result with the target
     contributing the most coordinate pairs removed.

Usage:
    python compute_statistics.py [path to stage0_analysis_workbook.xlsx]
    (defaults to ../results/stage0_analysis_workbook.xlsx)
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import openpyxl
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

THRESHOLD = 1.0
N_BOOTSTRAP = 10_000
RNG_SEED = 0


def load_raw_dimensions(workbook_path: Path):
    """Returns a list of (target, distance_signal, angle_signal) tuples,
    one per matched stem-pair coordinate dimension, read directly from
    the Finding1_RawData sheet's data rows."""
    wb = openpyxl.load_workbook(workbook_path, data_only=True)
    ws = wb["Finding1_RawData"]
    rows = []
    header_seen = False
    for row in ws.iter_rows(values_only=True):
        if row[0] == "Target":
            header_seen = True
            continue
        if not header_seen:
            continue
        if row[0] is None:
            break
        target, _, dist_signal, angle_signal = row[0], row[1], row[2], row[3]
        rows.append((target, float(dist_signal), float(angle_signal)))
    return rows


def target_level_delta(rows):
    by_target = defaultdict(lambda: {"dist": [], "angle": []})
    for target, dist, angle in rows:
        by_target[target]["dist"].append(dist)
        by_target[target]["angle"].append(angle)

    deltas = {}
    for target, vals in by_target.items():
        p_dist = np.mean([v > THRESHOLD for v in vals["dist"]])
        p_angle = np.mean([v > THRESHOLD for v in vals["angle"]])
        deltas[target] = p_angle - p_dist
    return deltas


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def pooled_mcnemar_and_wilson(rows, exclude_target=None):
    dist_clear = []
    angle_clear = []
    for target, dist, angle in rows:
        if exclude_target is not None and target == exclude_target:
            continue
        dist_clear.append(int(dist > THRESHOLD))
        angle_clear.append(int(angle > THRESHOLD))

    n = len(dist_clear)
    dist_clear = np.array(dist_clear)
    angle_clear = np.array(angle_clear)

    # McNemar's exact test on the paired (distance, angle) table
    both_clear = int(np.sum((dist_clear == 1) & (angle_clear == 1)))
    dist_only = int(np.sum((dist_clear == 1) & (angle_clear == 0)))
    angle_only = int(np.sum((dist_clear == 0) & (angle_clear == 1)))
    neither = int(np.sum((dist_clear == 0) & (angle_clear == 0)))
    table = [[both_clear, dist_only], [angle_only, neither]]
    result = mcnemar(table, exact=True)

    dist_count = int(dist_clear.sum())
    angle_count = int(angle_clear.sum())
    dist_ci = proportion_confint(dist_count, n, method="wilson")
    angle_ci = proportion_confint(angle_count, n, method="wilson")

    return {
        "n": n,
        "distance_count": dist_count,
        "distance_pct": dist_count / n,
        "angle_count": angle_count,
        "angle_pct": angle_count / n,
        "mcnemar_p": result.pvalue,
        "distance_wilson_ci": dist_ci,
        "angle_wilson_ci": angle_ci,
    }


def main():
    workbook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "results" / "stage0_analysis_workbook.xlsx"
    rows = load_raw_dimensions(workbook_path)
    print(f"Loaded {len(rows)} coordinate dimensions across "
          f"{len(set(r[0] for r in rows))} targets from {workbook_path}\n")

    # --- Target-level analysis ---
    deltas = target_level_delta(rows)
    delta_values = list(deltas.values())
    mean_delta = float(np.mean(delta_values))
    median_delta = float(np.median(delta_values))
    ci_lo, ci_hi = bootstrap_ci(delta_values)

    nonzero = [d for d in delta_values if d != 0]
    n_pos = sum(1 for d in nonzero if d > 0)
    sign_p = stats.binomtest(n_pos, len(nonzero), 0.5).pvalue if nonzero else float("nan")

    print("=== Target-level analysis ===")
    for target, d in sorted(deltas.items(), key=lambda kv: kv[1]):
        print(f"  {target}: Delta = {d:+.3f}")
    print(f"Mean Delta = {mean_delta:.3f}")
    print(f"Median Delta = {median_delta:.3f}")
    print(f"Bootstrap 95% CI on mean = [{ci_lo:.3f}, {ci_hi:.3f}] "
          f"({N_BOOTSTRAP} resamples, percentile method)")
    print(f"Exact sign test on {len(nonzero)} non-zero targets: p = {sign_p:.3f}\n")

    # --- Pooled-dimension analysis ---
    pooled = pooled_mcnemar_and_wilson(rows)
    print("=== Pooled-dimension analysis (full cohort) ===")
    print(f"n = {pooled['n']} pairs")
    print(f"Distance clearing: {pooled['distance_count']}/{pooled['n']} "
          f"({pooled['distance_pct']:.0%}), Wilson 95% CI "
          f"[{pooled['distance_wilson_ci'][0]:.0%}, {pooled['distance_wilson_ci'][1]:.0%}]")
    print(f"Angle clearing: {pooled['angle_count']}/{pooled['n']} "
          f"({pooled['angle_pct']:.0%}), Wilson 95% CI "
          f"[{pooled['angle_wilson_ci'][0]:.0%}, {pooled['angle_wilson_ci'][1]:.0%}]")
    print(f"McNemar's exact p = {pooled['mcnemar_p']:.4f}\n")

    # --- Leave-largest-contributor-out sensitivity ---
    counts = defaultdict(int)
    for target, _, _ in rows:
        counts[target] += 1
    largest = max(counts, key=counts.get)
    pooled_excl = pooled_mcnemar_and_wilson(rows, exclude_target=largest)
    print(f"=== Leave-one-target-out sensitivity (excluding {largest}, "
          f"the largest contributor with {counts[largest]} pairs) ===")
    print(f"n = {pooled_excl['n']} pairs")
    print(f"Distance clearing: {pooled_excl['distance_count']}/{pooled_excl['n']} "
          f"({pooled_excl['distance_pct']:.0%})")
    print(f"Angle clearing: {pooled_excl['angle_count']}/{pooled_excl['n']} "
          f"({pooled_excl['angle_pct']:.0%})")
    print(f"McNemar's exact p = {pooled_excl['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
