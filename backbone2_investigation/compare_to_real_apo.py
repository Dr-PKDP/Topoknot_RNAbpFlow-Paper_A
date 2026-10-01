"""
compare_to_real_apo.py -- the decisive test: does any Boltz-2 sample move
toward the REAL known apo conformation (5E54), not just scatter?

Real 5E54 apo C1' coordinates (PDB residues 19-37) are embedded directly
below -- fetched from files.rcsb.org/download/5E54.pdb. This range was
chosen deliberately: 5E54 and 1Y26 differ at 11 of 71 sequence positions,
but every difference is clustered at the two termini (1Y26 0-indexed
positions 0-5 and 65-70 -- see the sequence diff run earlier). PDB
residues 19-37 fall entirely inside the sequence-IDENTICAL stretch, so
any structural difference found here reflects real conformational
difference, not the confound of comparing two different sequences.

ASSUMPTION STATED EXPLICITLY, NOT VERIFIED: Boltz-2's auth_seq_id for a
target built from a bare YAML sequence (no explicit numbering given) is
assumed to be simple 1-indexed (residue 1 = first base in the sequence).
5E54's PDB numbering starts at 13 for that same first base (DBREF: '5E54
A 13 83'), so 5E54 residue N corresponds to Boltz-2 residue N-12. If this
assumption is wrong, every number below is offset by a fixed amount and
needs adjusting -- the script prints the actual residue count matched so
a systematic offset would show up as zero or near-zero overlap.

Usage:
    python compare_to_real_apo.py <dir with *_model_*.cif files>
"""
import glob
import sys
from pathlib import Path

import numpy as np

# Real 5E54 apo coordinates, PDB residue -> (x,y,z), chain A, C1' atom only.
# Fetched directly, not fabricated.
REAL_APO_5E54 = {
    19: (-14.192, 9.702, 12.310), 20: (-16.623, 9.657, 17.387),
    21: (-18.144, 7.147, 22.004), 22: (-11.962, 11.097, 27.590),
    23: (-7.969, 7.709, 24.658), 24: (-8.749, -1.512, 25.021),
    25: (-7.915, 4.737, 29.787), 26: (-6.672, 7.169, 34.662),
    27: (-5.322, 7.382, 40.010), 28: (-3.386, 4.817, 44.267),
    29: (-0.965, 1.080, 47.531), 30: (1.814, -3.234, 48.115),
    31: (5.224, -6.398, 45.497), 32: (9.411, -7.572, 42.279),
    33: (13.482, -6.832, 38.603), 34: (17.435, -4.027, 36.577),
    35: (19.996, -2.634, 29.900), 36: (19.671, -10.801, 29.241),
    37: (15.335, -6.094, 33.673),
}
PDB_TO_BOLTZ_OFFSET = 12  # 5E54 residue N == Boltz-2 residue N - 12


def read_c1_coords_cif(cif_path: Path) -> dict:
    coords = {}
    with open(cif_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            if len(parts) <= 12:
                continue
            atom_id = parts[3].strip('"')
            if atom_id != "C1'":
                continue
            resnum = int(parts[7])
            x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
            coords[resnum] = np.array([x, y, z])
    return coords


def kabsch_align(P: np.ndarray, Q: np.ndarray):
    """Returns (rmsd, P_aligned_onto_Q)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    aligned = (R @ Pc.T).T + Q.mean(0)
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - Q) ** 2, axis=1))))
    return rmsd, aligned


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    cif_files = sorted(glob.glob(str(Path(sys.argv[1]) / "*_model_*.cif")))
    if not cif_files:
        print(f"No *_model_*.cif files found in {sys.argv[1]}")
        sys.exit(1)

    real_apo_boltz_numbering = {
        r - PDB_TO_BOLTZ_OFFSET: np.array(xyz) for r, xyz in REAL_APO_5E54.items()
    }
    apo_resnums = sorted(real_apo_boltz_numbering.keys())

    print(f"Comparing {len(cif_files)} samples against real 5E54 apo coordinates")
    print(f"  (Boltz-2 residue numbers {apo_resnums[0]}-{apo_resnums[-1]}, "
          f"assumed offset {PDB_TO_BOLTZ_OFFSET} from 5E54 PDB numbering)\n")

    results = []
    for f in cif_files:
        sample_coords = read_c1_coords_cif(Path(f))
        matched = [r for r in apo_resnums if r in sample_coords]
        if len(matched) < len(apo_resnums):
            print(f"  WARNING [{Path(f).name}]: only {len(matched)}/{len(apo_resnums)} "
                  f"residues matched -- check the offset assumption")
        if len(matched) < 5:
            continue
        P = np.array([sample_coords[r] for r in matched])
        Q = np.array([real_apo_boltz_numbering[r] for r in matched])
        rmsd, _ = kabsch_align(P, Q)
        results.append((Path(f).name, rmsd, len(matched)))

    if not results:
        print("No samples had enough matched residues to compare -- "
              "the residue-numbering assumption is very likely wrong.")
        return

    results.sort(key=lambda r: r[1])
    print(f"{'sample':30s} {'RMSD to real apo (A)':>22s} {'residues matched':>18s}")
    for name, rmsd, n in results:
        print(f"{name:30s} {rmsd:22.2f} {n:18d}")

    rmsds = [r[1] for r in results]
    print(f"\nmean={np.mean(rmsds):.2f}  std={np.std(rmsds):.2f}  "
          f"min={np.min(rmsds):.2f}  max={np.max(rmsds):.2f}")


if __name__ == "__main__":
    main()
