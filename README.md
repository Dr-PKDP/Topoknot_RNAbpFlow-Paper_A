# Paper A — Generative RNA Structure Ensembles Recapitulate Experimentally Observed Conformational Variation

Single-backbone (RNAbpFlow) study testing whether generative RNA structure ensembles show
collective conformational variation aligned with real experimental structural states, across a
leakage-audited, tier-adjudicated cohort of RNA targets. Cross-backbone replication (H2) was
evaluated against three candidates and found unsupported; full reasoning in
`docs/PlanA_H2_outcome_draft.docx` (manuscript) and `docs/SUPPLEMENTARY_negative_results.docx`.

## Repository layout

```
data/cohort/          Target list and leakage-audit output (cohort.csv)
pipeline/              Core scripts: base-pair maps, pseudoknot detection, generation, convergence check
analysis/              Multi-modality diagnostic, K=50 vs K=100 comparison, synthetic validation,
                       case-study figures, statistics (compute_statistics.py)
backbone2_investigation/  Boltz-2 and pair-representation-scaling feasibility scripts (H2, closed)
docs/                  Manuscript and supplementary materials
figures/               Published figures (extracted from the manuscript; see figures/README.md)
results/               Final results workbook (data underlying Tables 2, 3, 5, 6, A1-A3)
```

## Reproducing a target from scratch

For any new target `<id>` with real sequence `<seq>`:

```bash
# 1. Base-pair maps (three-tool consensus)
cd pipeline
python3 generate_bp_maps.py <id> <seq> ../data/Inputs/<id>

# 2. Pseudoknot check -- always before generation, never after.
python3 generate_pseudoknot_pairs.py <id> <seq> ../data/Inputs/<id> <path-to-ipknot-binary>

# 3. Add <id> to the target list the generation script reads: this is
#    data/Inputs/list.txt, the path named in configs/inference_paperA_stage0.yaml's
#    input_dir -- not backbones/rnabpflow/Inputs/list.txt. Confirm the real file
#    directly before every run, and check the run's own startup line
#    ("Running Condition A... on N targets") against what is expected.

# 4. Generate (from backbones/rnabpflow/, with its own venv activated)
export CUDA_VISIBLE_DEVICES=1
python3 run_generative_only_inference_paperA.py

# 5. Analyze
cd ../pipeline
python3 stage0_pilot_check_v1.py --targets <id> \
  --sample-root ../data/Predictions --bpmaps-root ../data/Inputs \
  --n-perturbations 300 --out <id>_report.json

# 6. If K=50 CV is elevated (above ~12%), check for a real alternative
#    conformation before assuming more samples alone will resolve it:
cd ../analysis
python3 target_multimodality_diagnostic.py <id> ../data/Predictions/<id> "<dotbracket>"
```

## Environment reference

See `ENVIRONMENT.md`. Three conda environments are in use (`rnabpflow/venv`, `boltz2_check`,
`ipknot_build`) -- confirm with `which python3` before running anything; a stacked prompt is not
reliable confirmation of which environment's `python3` resolves first on `$PATH`.

## Cohort

14 targets in the primary, leakage-audited, tier-adjudicated cohort; see `data/cohort/`.

## Known issues affecting reproduction

1. **T01's originally-used representative (1Y26) is training-exposed.** The correct, clean
   primary-cohort representative is 5E54 (apo). 1Y26's results remain valid for pipeline
   validation and the (closed) backbone-2 investigation, never for the primary H1 claim.
2. **`list.txt` path**: the real file is wherever the YAML config's `input_dir` points,
   confirmed directly from source, not assumed from a plausible-looking path.
3. **RNA3DB training-exposure checks must filter to `split=='train_set'`** -- the full release
   also contains `test_set` rows; treating the combined file as "the training set" produces
   false positives.
4. **Pseudoknot checks are required before first generation for every target**, not a
   retroactive fix.

## Statistics

`analysis/compute_statistics.py` reads the raw per-dimension signal/floor values
directly from `results/stage0_analysis_workbook.xlsx` and computes the target-level
bootstrap CI, sign test, pooled McNemar's exact test, Wilson CIs, and the
leave-largest-target-out sensitivity check. This is a reconstruction built from the
procedure specified in the manuscript's Methods, not the original script recovered
from the server -- run it and compare against the manuscript's reported numbers
before relying on it; as of this writing its output matches every reported statistic
exactly (mean Delta = 0.036, 95% CI [-0.05, 0.15], pooled 13/29 and 15/29, Wilson CIs
[28%,62%] and [34%,69%], McNemar p = 0.69, leave-one-out 7/21 and 8/21, p = 1.00).
