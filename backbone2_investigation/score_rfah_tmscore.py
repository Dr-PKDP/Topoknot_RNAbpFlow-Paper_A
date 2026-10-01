"""
score_rfah_tmscore.py -- headless version of visualize_tmscore.py's scoring
cell, reusing its tm_to_ref() logic verbatim. Avoids running marimo
interactively on a headless remote server -- this is the real scoring
logic with a plain print-a-table wrapper instead of the plotting UI.

Usage (run from example/rfah/):
    python score_rfah_tmscore.py [output_dir]
    output_dir defaults to output_boltz2 if not given.
"""
import re
import sys
from pathlib import Path

from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data

SCRIPT = Path(__file__).parent
REF1 = SCRIPT / "data" / "ref" / "2OUG_C.pdb"   # alpha-helix CTD
REF2 = SCRIPT / "data" / "ref" / "6C6S_D.pdb"   # beta-barrel CTD
OUT_BOLTZ = SCRIPT / (sys.argv[1] if len(sys.argv) > 1 else "output_boltz2")


def parse_beta(dirname: str) -> float:
    m = re.match(r"beta_(neg|pos|zero)(?:(\d+)p(\d+))?", dirname)
    if m is None:
        return float("nan")
    sign, integer, frac = m.groups()
    if sign == "zero":
        return 0.0
    magnitude = float(f"{integer}.{frac}")
    return -magnitude if sign == "neg" else magnitude


def tm_to_ref(sample_path: Path, ref_path: Path) -> float:
    fmt = "mmcif" if sample_path.suffix == ".cif" else "pdb"
    s_chain = next(get_structure(str(sample_path), fmt).get_chains())
    r_chain = next(get_structure(str(ref_path)).get_chains())
    coords1, seq1 = get_residue_data(s_chain)
    coords2, seq2 = get_residue_data(r_chain)
    result = tm_align(coords1, coords2, seq1, seq2)
    return result.tm_norm_chain2


def main():
    if not REF1.exists() or not REF2.exists():
        raise FileNotFoundError(
            f"Reference structures not found -- expected {REF1} and {REF2}. "
            f"Check the example/rfah/data/ref/ directory actually shipped "
            f"with the repo."
        )

    rows = []
    for beta_dir in sorted(OUT_BOLTZ.iterdir()):
        if not beta_dir.is_dir():
            continue
        beta_value = parse_beta(beta_dir.name)
        samples = sorted(beta_dir.rglob("*_model_*.cif")) + sorted(beta_dir.rglob("*_model_*.pdb"))
        for s in samples:
            rows.append({
                "beta": beta_value,
                "sample": s.name,
                "tm_2OUG_C_helix": tm_to_ref(s, REF1),
                "tm_6C6S_D_barrel": tm_to_ref(s, REF2),
            })

    if not rows:
        print("No samples found under", OUT_BOLTZ)
        return

    rows.sort(key=lambda r: (r["beta"], r["sample"]))
    print(f"{'beta':>6s} {'sample':30s} {'TM to 2OUG_C (helix)':>22s} {'TM to 6C6S_D (barrel)':>22s}")
    for r in rows:
        print(f"{r['beta']:>6.2f} {r['sample']:30s} {r['tm_2OUG_C_helix']:>22.3f} {r['tm_6C6S_D_barrel']:>22.3f}")

    print("\nPer-beta means:")
    betas = sorted(set(r["beta"] for r in rows))
    for b in betas:
        subset = [r for r in rows if r["beta"] == b]
        mean_helix = sum(r["tm_2OUG_C_helix"] for r in subset) / len(subset)
        mean_barrel = sum(r["tm_6C6S_D_barrel"] for r in subset) / len(subset)
        print(f"  beta={b:+.2f}: mean TM-to-helix={mean_helix:.3f}  mean TM-to-barrel={mean_barrel:.3f}  (n={len(subset)})")


if __name__ == "__main__":
    main()
