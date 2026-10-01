# Environment reference

## Server access
```bash
ssh -J pduttapramanik@129.106.31.39 pduttapramanik@129.106.31.17
```
Always `which python3` immediately after activating any environment below -- a fresh shell can
silently remain in `base`, and a stacked prompt (e.g. `(venv) (boltz2_check)`) is not reliable
confirmation of which environment's `python3` actually resolves first on `$PATH`.

## `rnabpflow/venv` -- primary backbone, Stage 0/4 pipeline
```bash
cd /data/pduttapramanik/rna_uncertainty/backbones/rnabpflow
source venv/bin/activate
```
Used for: `generate_bp_maps.py`, `generate_pseudoknot_pairs.py`, `run_generative_only_inference_paperA.py`,
`stage0_pilot_check_v1.py`, `target_multimodality_diagnostic.py`, `compare_k50_vs_k100.py`.
Requires `torch_scatter` -- confirm with `python3 -c "import torch_scatter; print('ok')"` before
any generation run if there is any doubt the correct environment is active.

## `boltz2_check` (conda) -- Boltz-2 backbone-2 candidate (H2 investigation, closed)
```bash
conda activate boltz2_check
```
Used for: Boltz-2 inference (`boltz predict`, always with `--no_kernels` -- the
`cuequivariance_torch` package is not installed in this environment, and omitting the flag
produces a `ModuleNotFoundError` partway through the forward pass, after weights have already
loaded, which can look like a different failure if you're not expecting it). Also used for the
`pair-representation-scaling` (`prs`) CLI, installed via
`pip install -e ".[boltz-nokernels]"` inside this same environment (the tool runs on stock
`pip install boltz`, no vendored/patched fork needed as of the 2026/8/20 upstream release).

## `ipknot_build` (conda) -- pseudoknot detection
```bash
conda activate ipknot_build
```
Used for: building and running `generate_pseudoknot_pairs.py`'s IPknot dependency
(ViennaRNA + GLPK). IPknot binary path:
`/data/pduttapramanik/rna_uncertainty/ipknot/build/ipknot`.

## GPU discipline
Always `export CUDA_VISIBLE_DEVICES=1` before any GPU run. GPU 0 belongs to another
user/project on this shared server; confirm with `nvidia-smi` before running.
