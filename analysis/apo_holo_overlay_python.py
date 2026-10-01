"""
apo_holo_overlay.py -- pure-Python apo/holo structural-overlay figures,
no PyMOL required.

Install once:
    pip install biopython matplotlib numpy --break-system-packages

Run:
    python3 apo_holo_overlay.py

Downloads real coordinates from files.rcsb.org, superposes apo onto
holo using the Kabsch algorithm on matched C1' atoms (the same atom
used for this study's own collective coordinates, Methods S2.3), and
renders a backbone-trace overlay with the ligand shown.
"""

import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

CASE_STUDIES = [
    {
        "name": "TPP_riboswitch",
        "apo_id": "8F4O", "apo_chain": "X",
        "holo_id": "2GDI", "holo_chain": "X",
        "ligand_resn": "TPP",
    },
    {
        "name": "Adenine_riboswitch",
        "apo_id": "5E54", "apo_chain": "A",
        "holo_id": "4TZX", "holo_chain": "A",
        "ligand_resn": "ADE",  # verify against the actual structure --
                                 # ligand residue codes are not fully
                                 # standardized and should be checked
                                 # against the fetched file, not assumed
    },
]


def fetch_pdb(pdb_id, out_dir="."):
    path = f"{out_dir}/{pdb_id.lower()}.pdb"
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    urllib.request.urlretrieve(url, path)
    return path


def get_c1prime_by_resnum(structure, chain_id):
    """Returns {residue_number: np.array([x, y, z])} for every C1' atom
    in the given chain -- residue number is used as the correspondence
    key between apo and holo, which is robust to a differing number of
    total residues as long as the shared numbering scheme lines up
    (true for essentially all riboswitch apo/holo deposition pairs,
    since both are usually numbered from the same construct)."""
    coords = {}
    model = structure[0]
    chain = model[chain_id]
    for residue in chain:
        if "C1'" in residue:
            coords[residue.id[1]] = residue["C1'"].coord
    return coords


def kabsch_superpose(P, Q):
    """Returns the rotation matrix R and translation t that best
    superpose P onto Q (both N x 3 arrays of matched points), and the
    resulting RMSD. Standard SVD-based Kabsch algorithm."""
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean
    Q_c = Q - Q_mean
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = (R @ P_c.T).T + Q_mean
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
    return R, P_mean, Q_mean, rmsd, P_aligned


def build_overlay(case, out_dir="."):
    parser = PDBParser(QUIET=True)

    apo_path = fetch_pdb(case["apo_id"], out_dir)
    holo_path = fetch_pdb(case["holo_id"], out_dir)
    apo_struct = parser.get_structure(case["apo_id"], apo_path)
    holo_struct = parser.get_structure(case["holo_id"], holo_path)

    apo_coords = get_c1prime_by_resnum(apo_struct, case["apo_chain"])
    holo_coords = get_c1prime_by_resnum(holo_struct, case["holo_chain"])

    shared_resnums = sorted(set(apo_coords) & set(holo_coords))
    if len(shared_resnums) < 3:
        raise ValueError(
            f"[{case['name']}] only {len(shared_resnums)} matched C1' "
            f"positions found by residue number -- apo and holo chains "
            f"may use different numbering schemes; align manually or "
            f"adjust apo_chain/holo_chain."
        )

    P = np.array([apo_coords[r] for r in shared_resnums])   # apo, to move
    Q = np.array([holo_coords[r] for r in shared_resnums])  # holo, fixed
    R, P_mean, Q_mean, rmsd, P_aligned_matched = kabsch_superpose(P, Q)
    print(f"[{case['name']}] {len(shared_resnums)} matched C1' positions, "
          f"superposition RMSD = {rmsd:.2f} A")

    # apply the same transform to every apo C1' atom, not just the
    # matched subset, so the full backbone trace is shown post-alignment
    all_apo_resnums = sorted(apo_coords)
    P_full = np.array([apo_coords[r] for r in all_apo_resnums])
    P_full_aligned = (R @ (P_full - P_mean).T).T + Q_mean

    holo_resnums_sorted = sorted(holo_coords)
    Q_full = np.array([holo_coords[r] for r in holo_resnums_sorted])

    # ligand coordinates from the holo structure, if present
    ligand_coords = []
    for residue in holo_struct[0][case["holo_chain"]]:
        if residue.get_resname() == case["ligand_resn"]:
            for atom in residue:
                ligand_coords.append(atom.coord)
    if not ligand_coords:
        print(f"[{case['name']}] WARNING: no atoms matched resn "
              f"{case['ligand_resn']} in {case['holo_id']} chain "
              f"{case['holo_chain']} -- check the real HETATM residue "
              f"name in {holo_path} and update ligand_resn.")
    ligand_coords = np.array(ligand_coords) if ligand_coords else None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P_full_aligned[:, 0], P_full_aligned[:, 1], P_full_aligned[:, 2],
            "-o", color="#4472C4", markersize=3, linewidth=1.5,
            label=f"Apo ({case['apo_id']}, aligned)")
    ax.plot(Q_full[:, 0], Q_full[:, 1], Q_full[:, 2],
            "-o", color="#C0504D", markersize=3, linewidth=1.5,
            label=f"Holo ({case['holo_id']})")
    if ligand_coords is not None:
        ax.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2],
                   color="gold", s=40, label=case["ligand_resn"], depthshade=False)

    ax.set_title(f"{case['name'].replace('_', ' ')}: apo/holo C1' backbone "
                 f"overlay (RMSD = {rmsd:.2f} \u00c5, n={len(shared_resnums)})")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x (\u00c5)")
    ax.set_ylabel("y (\u00c5)")
    ax.set_zlabel("z (\u00c5)")
    plt.tight_layout()

    out_path = f"{out_dir}/{case['name']}_apo_holo_overlay.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[{case['name']}] wrote {out_path}")

    # ------------------------------------------------------------------
    # Stem/collective-coordinate annotation -- ADD YOUR OWN RESIDUE
    # RANGES HERE, from your actual pipeline's stem definitions. Example
    # once you have real stem boundaries (residue-number ranges):
    #
    #   stem1_apo_mid = P_full_aligned[[all_apo_resnums.index(r) for r in range(10, 16)]].mean(axis=0)
    #   stem2_apo_mid = P_full_aligned[[all_apo_resnums.index(r) for r in range(40, 46)]].mean(axis=0)
    #   dist_apo = np.linalg.norm(stem1_apo_mid - stem2_apo_mid)
    #   (repeat on Q_full for holo, and compare against the values
    #   already reported in Table 4 of the manuscript)
    # ------------------------------------------------------------------


for case in CASE_STUDIES:
    build_overlay(case)
