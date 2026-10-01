"""
boltz_diversity_check.py -- raw vs. superposed RMSD across a Boltz-2
multi-sample prediction, directly comparable to the same check already
run on RNAbpFlow's output (stage0_pilot_check_v1.py). Same two functions
(raw_rmsd, kabsch_rmsd), same logic -- only the coordinate-reading
function differs, since Boltz-2 outputs mmCIF, not legacy PDB.

Column layout confirmed directly from a real Boltz-2 output file's own
mmCIF header (not assumed from the general mmCIF spec, which allows
different column orders):
  group_PDB id type_symbol label_atom_id label_alt_id label_comp_id
  label_seq_id auth_seq_id pdbx_PDB_ins_code label_asym_id
  Cartn_x Cartn_y Cartn_z occupancy label_entity_id auth_asym_id
  auth_comp_id B_iso_or_equiv pdbx_PDB_model_num

Atom names containing an apostrophe (C1', C4', O5', ...) are written
quoted in the raw file, e.g. "C1'" -- the quote characters are literal
text in the line, not something a plain whitespace split removes, so
they are stripped explicitly here.

Usage:
    python boltz_diversity_check.py <dir with *_model_*.cif files>
"""
import glob
import sys
from pathlib import Path

import numpy as np

# column indices, 0-indexed, confirmed against a real Boltz-2 output file
COL_ATOM_ID = 3
COL_AUTH_SEQ_ID = 7
COL_X, COL_Y, COL_Z = 10, 11, 12


def read_c1_coords_cif(cif_path: Path) -> dict:
    """Returns {auth_seq_id (int): xyz} for every C1' atom in the file."""
    coords = {}
    with open(cif_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            if len(parts) <= COL_Z:
                continue
            atom_id = parts[COL_ATOM_ID].strip('"')
            if atom_id != "C1'":
                continue
            resnum = int(parts[COL_AUTH_SEQ_ID])
            x, y, z = float(parts[COL_X]), float(parts[COL_Y]), float(parts[COL_Z])
            coords[resnum] = np.array([x, y, z])
    return coords


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    aligned = (R @ Pc.T).T
    return float(np.sqrt(np.mean(np.sum((aligned - Qc) ** 2, axis=1))))


def raw_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    cif_files = sorted(glob.glob(str(Path(sys.argv[1]) / "*_model_*.cif")))
    if not cif_files:
        print(f"No *_model_*.cif files found in {sys.argv[1]}")
        sys.exit(1)

    all_coords = [read_c1_coords_cif(Path(f)) for f in cif_files]
    lens = set(len(c) for c in all_coords)
    print(f"{len(cif_files)} structures, C1' atom counts per structure: {lens}")
    if len(lens) > 1:
        print("  WARNING: inconsistent atom counts across samples -- check parsing")

    keys_common = sorted(set.intersection(*[set(c.keys()) for c in all_coords]))
    print(f"  {len(keys_common)} residues common to all samples")
    coord_mats = [np.array([c[k] for k in keys_common]) for c in all_coords]

    n = len(coord_mats)
    raw_list, sup_list = [], []
    per_sample_sup = {i: [] for i in range(n)}
    for a in range(n):
        for b in range(a + 1, n):
            r = raw_rmsd(coord_mats[a], coord_mats[b])
            s = kabsch_rmsd(coord_mats[a], coord_mats[b])
            raw_list.append(r)
            sup_list.append(s)
            per_sample_sup[a].append(s)
            per_sample_sup[b].append(s)
    raw_arr, sup_arr = np.array(raw_list), np.array(sup_list)

    print(f"\npairwise RAW RMSD:        mean={raw_arr.mean():.2f}  std={raw_arr.std():.2f}  "
          f"min={raw_arr.min():.2f}  max={raw_arr.max():.2f}  (n={len(raw_arr)} pairs)")
    print(f"pairwise SUPERPOSED RMSD: mean={sup_arr.mean():.2f}  std={sup_arr.std():.2f}  "
          f"min={sup_arr.min():.2f}  max={sup_arr.max():.2f}")
    global_frac = 1.0 - (sup_arr.mean() / raw_arr.mean()) if raw_arr.mean() > 0 else None
    print(f"global rigid-body fraction: {global_frac:.3f}" if global_frac is not None else
          "global rigid-body fraction: undefined (raw RMSD is zero)")

    print("\nper-sample mean superposed RMSD to all other samples "
          "(cross-reference against each sample's confidence_score to check "
          "whether low-confidence samples are also the geometric outliers -- "
          "if so, diversity is more likely degradation than real alternate states):")
    for i, f in enumerate(cif_files):
        mean_to_others = float(np.mean(per_sample_sup[i]))
        print(f"  {Path(f).name}: mean RMSD to others = {mean_to_others:.2f} A")


if __name__ == "__main__":
    main()
