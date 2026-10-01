# Figures

Extracted directly from the published manuscript (`docs/PlanA_H2_outcome_draft.docx`),
not regenerated. These are the exact images that appear in the paper.

- `fig1_cohort_flow.png` — cohort construction flow, candidate pool through the
  9-target analyzable subset
- `fig2_topology_calibration.png` — topology-aware perturbation calibration (4 panels)
- `fig3_target_delta.png` — target-level Delta forest plot with bootstrap CI
- `fig4_case_studies.png` — TPP and adenine riboswitch case studies
- `figS5_magnitude_comparison.png` — model signal vs. published transition magnitude

The matplotlib scripts that generated these are not in this repo -- they were built
in a separate working session, not on the GPU server, and are not recoverable from
here. The images above are the real, exact published output; regenerating equivalent
plots from the data in `results/stage0_analysis_workbook.xlsx` and
`data/cohort/paperA_cohort_candidates_FINAL.csv` is possible but would not be
guaranteed to match these pixel-for-pixel.
