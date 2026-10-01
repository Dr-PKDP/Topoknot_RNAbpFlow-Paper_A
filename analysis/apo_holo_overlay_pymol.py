"""
apo_holo_overlay.py -- PyMOL script for real apo/holo structural-overlay
case-study figures (TPP riboswitch: 8F4O vs 2GDI; adenine riboswitch:
5E54 vs 4TZX).

Run with: pymol -cq apo_holo_overlay.py
(or paste into an interactive PyMOL session)

Requires: PyMOL (open-source or commercial), internet access to
files.rcsb.org. No other dependencies.
"""

from pymol import cmd

# ----------------------------------------------------------------------
# Config -- one entry per case study. Edit CHAIN if you want a specific
# chain from a multi-chain deposition (8F4O and 2GDI both have chains
# X and Y as two crystallographically independent copies; chain X is
# used below for both, arbitrarily but consistently).
# ----------------------------------------------------------------------
CASE_STUDIES = [
    {
        "name": "TPP_riboswitch",
        "apo_id": "8f4o", "apo_chain": "X",
        "holo_id": "2gdi", "holo_chain": "X",
        "ligand_resn": "TPP",
    },
    {
        "name": "Adenine_riboswitch",
        "apo_id": "5e54", "apo_chain": "A",
        "holo_id": "4tzx", "holo_chain": "A",
        "ligand_resn": "ADE",  # verify against the actual HETATM records
                                 # after fetching -- ligand residue names
                                 # are not fully standardized across PDB
                                 # depositions and this should be checked,
                                 # not assumed.
    },
]

OUTDIR = "."  # change to your preferred output directory


def build_overlay(case):
    apo_obj = f"{case['apo_id']}_apo"
    holo_obj = f"{case['holo_id']}_holo"

    cmd.reinitialize()
    cmd.fetch(case["apo_id"], name=apo_obj, type="pdb")
    cmd.fetch(case["holo_id"], name=holo_obj, type="pdb")

    # keep only the chosen chain of each, plus any bound ligand/ions on
    # that chain -- drops the second crystallographic copy and solvent
    cmd.remove(f"{apo_obj} and not chain {case['apo_chain']}")
    cmd.remove(f"{holo_obj} and not chain {case['holo_chain']}")
    cmd.remove("solvent")

    # structure-based alignment (sequence-independent), appropriate here
    # since apo and holo constructs sometimes differ slightly in
    # numbering or a few flanking residues. cmd.super is generally more
    # robust than cmd.align for RNA-RNA fits; switch to cmd.align if you
    # specifically want a sequence-guided fit instead.
    rms = cmd.super(f"{holo_obj} and polymer", f"{apo_obj} and polymer")
    print(f"[{case['name']}] structural superposition RMSD: {rms[0]:.2f} A "
          f"over {rms[1]} atom pairs")

    # representation
    cmd.hide("everything")
    cmd.show("cartoon", f"{apo_obj} or {holo_obj}")
    cmd.set("cartoon_ring_mode", 3)
    cmd.set("cartoon_ring_finder", 1)
    cmd.color("skyblue", apo_obj)
    cmd.color("salmon", holo_obj)

    # ligand, if present in the holo structure
    lig_sel = f"{holo_obj} and resn {case['ligand_resn']}"
    if cmd.count_atoms(lig_sel) > 0:
        cmd.show("sticks", lig_sel)
        cmd.color("yellow", lig_sel)
    else:
        print(f"[{case['name']}] WARNING: no atoms matched resn "
              f"{case['ligand_resn']} in {holo_obj} -- check the actual "
              f"HETATM residue name in the fetched file and update "
              f"ligand_resn above.")

    # ------------------------------------------------------------------
    # Stem/collective-coordinate annotation -- ADD YOUR OWN RESIDUE
    # RANGES HERE. The paper's collective coordinates are defined per
    # stem pair from C1' atom coordinates (Methods S2.3); this script
    # does not know your pipeline's specific stem boundary definitions.
    # Example of how to add a distance measurement between two stem
    # midpoints once you have the real residue ranges:
    #
    #   cmd.pseudoatom("stem1_apo_mid", selection=f"{apo_obj} and resi 10-15 and name C1'")
    #   cmd.pseudoatom("stem2_apo_mid", selection=f"{apo_obj} and resi 40-45 and name C1'")
    #   cmd.distance("d1_apo", "stem1_apo_mid", "stem2_apo_mid")
    #   (repeat for the holo structure to compare)
    # ------------------------------------------------------------------

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.orient(f"{apo_obj} or {holo_obj}")
    cmd.zoom(f"{apo_obj} or {holo_obj}", buffer=5)

    png_path = f"{OUTDIR}/{case['name']}_apo_holo_overlay.png"
    cmd.ray(1600, 1600)
    cmd.png(png_path, dpi=300)
    print(f"[{case['name']}] wrote {png_path}")

    # also save the aligned session for further work in the GUI
    cmd.save(f"{OUTDIR}/{case['name']}_apo_holo_overlay.pse")


for case in CASE_STUDIES:
    build_overlay(case)
