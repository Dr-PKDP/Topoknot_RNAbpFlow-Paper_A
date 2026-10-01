"""
compare_k50_vs_k100.py -- two things, done precisely rather than by eye:
1. Prints each target's full K-convergence curve (CV% at every K checked),
   to see whether uncertainty keeps shrinking past K=50 or plateaus.
2. Diffs the new K=100 signal/floor numbers against the original K=50
   pilot numbers (hardcoded below, exactly as reported in the Stage 0
   write-up) to check whether Finding 1 (angle clears the floor more
   often than distance, ~15x median gap) still holds.

Usage:
    python compare_k50_vs_k100.py stage0_report_k100.json
"""
import json
import sys

# Original K=50 pilot numbers, exactly as reported in Stage0_pilot_results.
ORIGINAL_K50 = {
    "1y26": {"dist": [0.37, 0.44, 0.52], "angle": [18.17, 0.21, 73.53]},
    "2miy": {"dist": [1.95], "angle": [27.59]},
    "4rzd": {"dist": [0.06, 0.22], "angle": [0.74, 0.42]},
    "7kd1": {"dist": [1.03, 1.47, 0.02], "angle": [0.27, 6.85, 0.02]},
    "5kh8": {"dist": [0.24, 8.17], "angle": [12.37, 14.68]},
}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    report = json.load(open(sys.argv[1]))

    print("=" * 70)
    print("PART 1: K-convergence curves (CV% at each K -- lower is more converged)")
    print("=" * 70)
    for t in ["1y26", "2miy", "4rzd", "7kd1", "5kh8"]:
        if t not in report or "k_convergence" not in report[t]:
            print(f"{t}: no k_convergence data found in report")
            continue
        kc = report[t]["k_convergence"]
        print(f"\n{t}:")
        for k in sorted(kc, key=int):
            m = kc[k].get("mean_frob_norm")
            s = kc[k].get("std_frob_norm")
            if m is None or s is None:
                continue
            cv = 100 * s / m if m else 0.0
            print(f"  K={int(k):>3d}: mean_frob_norm={m:8.2f}  CV={cv:5.1f}%")

    print()
    print("=" * 70)
    print("PART 2: K=50 (original) vs K=100 (this run) signal/floor comparison")
    print("=" * 70)
    for t in ["1y26", "2miy", "4rzd", "7kd1", "5kh8"]:
        if t not in report:
            continue
        new_dist = report[t].get("signal_to_floor_ratio_distance", [])
        new_angle = report[t].get("signal_to_floor_ratio_angle", [])
        old = ORIGINAL_K50.get(t, {"dist": [], "angle": []})
        print(f"\n{t}:")
        print(f"  distance  K=50 (orig): {old['dist']}")
        print(f"  distance  K=100 (new): {[round(float(x), 2) for x in new_dist]}")
        print(f"  angle     K=50 (orig): {old['angle']}")
        print(f"  angle     K=100 (new): {[round(float(x), 2) for x in new_angle]}")


if __name__ == "__main__":
    main()
