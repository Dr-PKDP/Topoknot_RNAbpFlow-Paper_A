"""
Stage 0 pilot check -- runs the go/no-go diagnostic specified in Plan A v3
Appendix A, on top of collective_coords_v1.py.

noise_floor's perturbation-attempt count is a CLI flag
(--n-perturbations, default 100). This allows a target that falls short
of the 10-valid-perturbation threshold to be rerun with more attempts
using the same already-generated samples -- no new GPU inference is
needed, since the noise floor is computed from fixed sample coordinates
and only the secondary-structure perturbation count changes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from collective_coords_v1 import (extract_z, parse_dotbracket, find_stems, Stem,
                                   nested_pairs_only, mark_pseudoknot_stems)


def read_c1_coords(pdb_path: Path) -> dict:
    coords = {}
    seen_res_keys = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "C1'":
                continue
            chain = line[21]
            resseq = line[22:26].strip()
            icode = line[26]
            res_key = (chain, resseq, icode)
            if res_key in seen_res_keys:
                continue
            seen_res_keys.append(res_key)
            idx = len(seen_res_keys) - 1
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            coords[idx] = np.array([x, y, z])
    return coords


def load_consensus_structure(target_name: str, structure_dir: Path,
                              prob_threshold: float = 0.5) -> str:
    """Derives one dot-bracket secondary structure for a target.

    Two real, different outcomes were observed testing this against the
    pilot cohort, not one -- and they call for different handling:

      - When the three-tool (MXfold2/SimFold/CONTRAfold) nested consensus
        genuinely has nothing to say about a region (5KH8: residues 6-9
        and 40-43 were unpaired dots), IPknot's crossing pairs slot in
        cleanly with no conflict.
      - When the three-tool consensus confidently but WRONGLY resolves a
        pseudoknotted region into its own nested-only guess (4RZD:
        residues 56-77 paired at 0.67 probability across all three tools,
        an answer nearly identical to IPknot's own crossing-arm residues,
        but missing the embedded inner stem entirely, since three
        non-pseudoknot-aware tools folding a crossing region can't do
        better than the closest non-crossing approximation), a
        residue-level merge with the nested consensus taking priority on
        collision throws the correct topology away rather than adding to
        it -- confirmed directly: 4RZD's merge accepted 0 of 7 real
        crossing pairs.

    So: once IPknot confirms a target genuinely contains a crossing
    structure (i.e. it returned at least one crossing pair), this
    function uses IPknot's own COMPLETE prediction (both its nested and
    crossing components) for the whole target, rather than attempting a
    residue-level splice against the three-tool consensus. RNA folding
    energetics are global, not local -- three tools constrained to a
    nested-only answer may distort their guess for the entire sequence
    accommodating a crossing they cannot represent, not just at the
    crossing site, so trusting one coherent crossing-aware prediction is
    more defensible than stitching two different models together with an
    arbitrary priority rule at the seams.

    Targets where IPknot finds NO crossing pairs (2MIY: zero found across
    four threshold/model settings, consistent with its published
    ligand-induced pseudoknot) fall back to the three-tool consensus
    exactly as before -- there is no evidence here that IPknot's nested-
    only call for such a target is better than three tools' agreement,
    so nothing changes for this case.
    """
    pk_path = structure_dir / "pseudoknot_pairs.json"
    if pk_path.exists():
        pk_data = json.loads(pk_path.read_text())
        crossing = pk_data.get("crossing_pairs", [])
        if crossing:
            db = pk_data["ipknot_dotbracket"]
            print(f"  [{target_name}] confirmed pseudoknot ({len(crossing)} crossing "
                  f"pairs) -- using IPknot's complete prediction for this target, "
                  f"not the three-tool consensus", file=sys.stderr)
            return db

    maps = []
    for i in (1, 2, 3):
        p = structure_dir / f"map{i}.npy"
        if p.exists():
            maps.append(np.load(p))
    if not maps:
        raise FileNotFoundError(
            f"no map1/2/3.npy found in {structure_dir} for {target_name!r}")
    L = maps[0].shape[0]
    for m in maps:
        if m.shape != (L, L):
            raise ValueError(f"map shape mismatch for {target_name!r}: {m.shape} vs {(L, L)}")

    mean_prob = np.mean(maps, axis=0)
    mean_prob = (mean_prob + mean_prob.T) / 2.0

    candidates = []
    for i in range(L):
        for j in range(i + 1, L):
            p = mean_prob[i, j]
            if p >= prob_threshold:
                candidates.append((p, i, j))
    candidates.sort(reverse=True)

    used = set()
    nested_pairs = []
    for p, i, j in candidates:
        if i in used or j in used:
            continue
        nested_pairs.append((i, j))
        used.add(i)
        used.add(j)

    db = ["."] * L
    for i, j in nested_pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = (R @ Pc.T).T + Q.mean(0)
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))
    return rmsd, R, Q.mean(0) - R @ P.mean(0)


def raw_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def perturb_dotbracket(db: str, rng: np.random.Generator, max_shift=1):
    pairs = parse_dotbracket(db)
    stems = find_stems(pairs)
    # BUGFIX: mark_pseudoknot_stems was never called here, so every stem's
    # is_pseudoknot silently defaulted to False -- every kept pair got written
    # back as a plain nested pair ("(", ")") below, regardless of whether it
    # originally belonged to a pseudoknot-crossing stem. This meant every
    # perturbation of a pseudoknotted target silently discarded its crossing
    # pairs, causing near-total failure of noise_floor()'s length check for
    # pseudoknotted targets (observed directly: 0-3 valid perturbations out
    # of 300 for T07/T09, versus 90-300 for non-pseudoknotted targets).
    nested_set = set(nested_pairs_only(pairs))
    mark_pseudoknot_stems(stems, nested_set)
    n = len(db)

    kept = []  # list of (i, j, is_pk) so the correct bracket type can be written below
    for stem in stems:
        stem_pairs = list(stem.pairs)
        is_pk = stem.is_pseudoknot
        if len(stem_pairs) == 1:
            i, j = stem_pairs[0]
            kept.append((i, j, is_pk))
            continue
        outer_i, outer_j = stem_pairs[0]
        inner_i, inner_j = stem_pairs[-1]

        shift_outer = int(rng.integers(-max_shift, max_shift + 1))
        new_outer_i = np.clip(outer_i + shift_outer, 0, inner_i - 1)
        new_outer_j = np.clip(outer_j - shift_outer, inner_j + 1, n - 1)

        shift_inner = int(rng.integers(-max_shift, max_shift + 1))
        new_inner_i = np.clip(inner_i + shift_inner, new_outer_i + 1, n - 1)
        new_inner_j = np.clip(inner_j - shift_inner, 0, new_outer_j - 1)
        if new_inner_i >= new_inner_j:
            new_inner_i, new_inner_j = inner_i, inner_j

        middle = stem_pairs[1:-1]
        kept.append((int(new_outer_i), int(new_outer_j), is_pk))
        kept.extend((i, j, is_pk) for (i, j) in middle)
        kept.append((int(new_inner_i), int(new_inner_j), is_pk))

    new_db = ["."] * n
    used_positions = set()
    for (i, j, is_pk) in kept:
        if i in used_positions or j in used_positions or i >= j:
            continue
        new_db[i] = "[" if is_pk else "("
        new_db[j] = "]" if is_pk else ")"
        used_positions.add(i)
        used_positions.add(j)
    return "".join(new_db)


def noise_floor(coords: dict, db: str, n_perturbations=100, seed=0):
    rng = np.random.default_rng(seed)
    zs = []
    base_z, base_labels = extract_z(coords, db)
    for _ in range(n_perturbations):
        pdb2 = perturb_dotbracket(db, rng)
        try:
            z2, labels2 = extract_z(coords, pdb2)
        except Exception:
            continue
        if len(z2) == len(base_z):
            zs.append(z2)
    if len(zs) < 10:
        return None, base_labels, len(zs)
    zs = np.array(zs)
    return zs.var(axis=0), base_labels, len(zs)


def run_target(target_name: str, sample_dir: Path, dotbracket: str, out: dict,
                n_perturbations: int = 100):
    sample_files = sorted(sample_dir.glob("Sample_*.pdb"))
    if not sample_files:
        raise FileNotFoundError(f"no Sample_*.pdb files found in {sample_dir}")
    K = len(sample_files)

    all_coords = []
    z_list = []
    labels = None
    for f in sample_files:
        coords = read_c1_coords(f)
        if len(coords) != len(dotbracket):
            print(f"  WARNING [{target_name}/{f.name}]: {len(coords)} C1' atoms read, "
                  f"{len(dotbracket)} positions in structure -- indexing may be off",
                  file=sys.stderr)
        all_coords.append(coords)
        z, labels = extract_z(coords, dotbracket)
        z_list.append(z)

    z_arr = np.array(z_list)
    n_dist = sum(1 for l in labels if l.startswith("d["))
    n_ang = sum(1 for l in labels if l.startswith("theta["))

    keys_common = sorted(set.intersection(*[set(c.keys()) for c in all_coords]))
    coord_mats = [np.array([c[k] for k in keys_common]) for c in all_coords]
    raw_rmsds, sup_rmsds = [], []
    for a in range(K):
        for b in range(a + 1, K):
            raw_rmsds.append(raw_rmsd(coord_mats[a], coord_mats[b]))
            r, _, _ = kabsch_rmsd(coord_mats[a], coord_mats[b])
            sup_rmsds.append(r)
    raw_mean, sup_mean = float(np.mean(raw_rmsds)), float(np.mean(sup_rmsds))

    z_var_dist = z_arr[:, :n_dist].var(axis=0) if n_dist else np.array([])
    z_var_ang = z_arr[:, n_dist:].var(axis=0) if n_ang else np.array([])

    floor_var, floor_labels, n_ok = noise_floor(all_coords[0], dotbracket,
                                                  n_perturbations=n_perturbations)
    floor_dist = floor_var[:n_dist] if floor_var is not None and n_dist else None
    floor_ang = floor_var[n_dist:] if floor_var is not None and n_ang else None

    convergence = {}
    ks_to_check = [k for k in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) if k <= K]
    rng = np.random.default_rng(42)
    for k in ks_to_check:
        norms = []
        for _ in range(50):
            idx = rng.choice(K, size=k, replace=False)
            cov = np.cov(z_arr[idx].T)
            norms.append(np.linalg.norm(cov, "fro"))
        convergence[k] = {"mean_frob_norm": float(np.mean(norms)),
                           "std_frob_norm": float(np.std(norms))}

    out[target_name] = {
        "K": K,
        "n_distance_coords": n_dist,
        "n_angle_coords": n_ang,
        "raw_rmsd_mean": raw_mean,
        "superposed_rmsd_mean": sup_mean,
        "global_rigid_body_fraction": 1.0 - (sup_mean / raw_mean) if raw_mean > 0 else None,
        "z_variance_distance_block": z_var_dist.tolist(),
        "z_variance_angle_block": z_var_ang.tolist(),
        "noise_floor_distance_block": floor_dist.tolist() if floor_dist is not None else None,
        "noise_floor_angle_block": floor_ang.tolist() if floor_ang is not None else None,
        "noise_floor_n_valid_perturbations": n_ok,
        "noise_floor_n_perturbations_attempted": n_perturbations,
        "signal_to_floor_ratio_distance": (
            [(round(s / f, 2) if f > 1e-6 else "floor near-zero, ratio not meaningful")
             for s, f in zip(z_var_dist, floor_dist)]
            if floor_dist is not None and n_dist else
            (f"not computed -- only {n_ok} valid perturbations survived "
             f"(need >= 10)" if n_dist else None)
        ),
        "signal_to_floor_ratio_angle": (
            [(round(s / f, 2) if f > 1e-6 else "floor near-zero, ratio not meaningful")
             for s, f in zip(z_var_ang, floor_ang)]
            if floor_ang is not None and n_ang else
            (f"not computed -- only {n_ok} valid perturbations survived "
             f"(need >= 10)" if n_ang else None)
        ),
        "k_convergence": convergence,
        "coordinate_labels": labels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--sample-root", type=Path, required=True)
    ap.add_argument("--bpmaps-root", type=Path, default=None)
    ap.add_argument("--dotbracket-file", type=Path, default=None)
    ap.add_argument("--n-perturbations", type=int, default=100,
                     help="perturbation attempts for the noise floor (raise this "
                          "for a target that reported too few valid perturbations "
                          "at the default -- reruns cheaply, no new GPU inference "
                          "needed, since it only re-perturbs the secondary structure "
                          "against already-generated sample coordinates)")
    ap.add_argument("--out", type=Path, default=Path("stage0_report.json"))
    args = ap.parse_args()

    if args.dotbracket_file:
        db_map = json.loads(args.dotbracket_file.read_text())
    else:
        db_map = None

    if not args.dotbracket_file and not args.bpmaps_root:
        ap.error("--bpmaps-root is required unless --dotbracket-file is given")

    out = {}
    for t in args.targets:
        db = db_map[t] if db_map else load_consensus_structure(t, args.bpmaps_root / t)
        print(f"Running {t} ...")
        run_target(t, args.sample_root / t, db, out, n_perturbations=args.n_perturbations)
        r = out[t]
        print(f"  K={r['K']}  raw/superposed RMSD = {r['raw_rmsd_mean']:.2f} / "
              f"{r['superposed_rmsd_mean']:.2f} A  "
              f"(global rigid-body fraction = {r['global_rigid_body_fraction']:.2f})")
        print(f"  valid perturbations: {r['noise_floor_n_valid_perturbations']}/"
              f"{r['noise_floor_n_perturbations_attempted']} attempted")
        if r["signal_to_floor_ratio_distance"]:
            print(f"  distance-block signal/floor: {r['signal_to_floor_ratio_distance']}")
        if r["signal_to_floor_ratio_angle"]:
            print(f"  angle-block signal/floor: {r['signal_to_floor_ratio_angle']}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
