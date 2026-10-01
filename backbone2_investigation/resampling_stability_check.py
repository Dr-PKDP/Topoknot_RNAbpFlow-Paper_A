"""
resampling_stability_check.py -- for a target whose noise-floor calibration
used a small number of accepted perturbations (e.g. T07's 27/300), checks
whether the resulting signal-to-floor result is stable under resampling, or
whether it's sensitive to exactly which small subset of perturbations got
used -- the concrete diagnostic a third-party review proposed for exactly
this situation.

Re-runs the SAME perturbation process as stage0_pilot_check_v1.py's
noise_floor() (same seed, same logic, via the same imported functions --
not duplicated/reimplemented), but keeps every accepted perturbation's raw
z-vector instead of collapsing straight to a single variance. Then, for
each subsample size in a given list, draws many random subsamples of that
size from the accepted set and reports the distribution of the resulting
signal-to-floor ratio -- if the ratio swings wildly as n shrinks, the
original result is not yet trustworthy at its reported magnitude; if it
stays roughly stable, the result is robust.

Usage:
    python resampling_stability_check.py <target> <sample_dir> <dotbracket> <n_perturbations>
"""
import sys
from pathlib import Path

import numpy as np

from stage0_pilot_check_v1 import (perturb_dotbracket, read_c1_coords)
from collective_coords_v1 import extract_z


def collect_accepted_perturbations(coords, db, n_perturbations, seed=0):
    """Same logic as noise_floor(), but returns every accepted z-vector,
    not just their collapsed variance."""
    rng = np.random.default_rng(seed)
    base_z, base_labels = extract_z(coords, db)
    zs = []
    for _ in range(n_perturbations):
        pdb2 = perturb_dotbracket(db, rng)
        try:
            z2, _ = extract_z(coords, pdb2)
        except Exception:
            continue
        if len(z2) == len(base_z):
            zs.append(z2)
    return np.array(zs), base_labels


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    target, sample_dir, db, n_perturbations = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])

    sample_files = sorted(sample_dir.glob("Sample_*.pdb"))
    if not sample_files:
        raise FileNotFoundError(f"no Sample_*.pdb files found in {sample_dir}")
    # use the first sample's coordinates, matching noise_floor()'s own convention
    # of computing the perturbation baseline from a single reference structure
    coords = read_c1_coords(sample_files[0])

    zs, base_labels = collect_accepted_perturbations(coords, db, n_perturbations)
    n_accepted = len(zs)
    print(f"{target}: {n_accepted}/{n_perturbations} perturbations accepted "
          f"(same seed=0 as the main pipeline -- should match the already-reported count)")
    if n_accepted < 10:
        print("Fewer than 10 accepted -- resampling stability cannot be assessed "
              "(this is the T09-type case, not the T07-type case).")
        return

    n_dist = sum(1 for l in base_labels if l.startswith("d["))
    subsample_sizes = [n for n in (10, 15, 20, 25) if n <= n_accepted]
    n_bootstrap = 200
    rng = np.random.default_rng(1)

    print(f"\nFull-sample (n={n_accepted}) floor variance per dimension:")
    full_var = zs.var(axis=0)
    print("  " + ", ".join(f"{v:.4f}" for v in full_var))

    print(f"\nResampling stability (n_bootstrap={n_bootstrap} draws per subsample size):")
    for n in subsample_sizes:
        boot_vars = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_accepted, size=n, replace=False)
            boot_vars.append(zs[idx].var(axis=0))
        boot_vars = np.array(boot_vars)  # shape (n_bootstrap, n_dims)
        mean_var = boot_vars.mean(axis=0)
        cv_across_bootstrap = 100 * boot_vars.std(axis=0) / np.where(mean_var == 0, np.nan, mean_var)
        print(f"  n={n:>3d}: mean floor variance per dim = "
              + ", ".join(f"{v:.4f}" for v in mean_var))
        print(f"         variability across resamples (CV%) = "
              + ", ".join(f"{v:.0f}%" if not np.isnan(v) else "n/a" for v in cv_across_bootstrap))

    print("\nHow to read this: compare the full-sample floor variance (n={}) against each "
          "subsample size's mean. If they stay close and the resample-to-resample CV% is "
          "modest (roughly under 50%), the original result is stable. If the mean drifts "
          "substantially as n shrinks, or the CV% is very high (multiple hundred percent), "
          "the original n={}-based result should be treated as provisional, not confirmed."
          .format(n_accepted, n_accepted))


if __name__ == "__main__":
    main()
