# Stage 4 reproduction and scope

This is a bounded exploratory follow-up to the Stage 3 negative intervention
result. It uses already-observed Stage 3 scenarios and does not constitute an
independent confirmation set. The locally frozen plan is `stage4-preregistration.md`.

## Read-only numerical audit

From the repository root, with Python >=3.11 and NumPy:

```powershell
python scripts/audit_stage4.py
```

This independently checks reference/source hashes, donor pairing, input-position
alignment, fixed train-only NF4 directions, displacement norms, delivered full
patch norms, raw effects, scenario bootstrap intervals, paired comparisons,
categorical changes, generation/null checks and artifact manifests. It does not
import the Stage 4 summary implementation, torch, or sklearn. A successful audit
is not a fresh model-inference replication. It requires the unchanged Stage 3
reference baseline, prompt and activation files as well as Stage 4 outputs.

## Inference commands actually used

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
python -u -m intent_conflict.stage4 --precision nf4
python -u -m intent_conflict.stage4 --precision bf16
```

Do not rerun these into existing completed directories: the runner refuses to
overwrite them. Use a separate workspace copy or explicitly version a new output
root/config. Do not delete or relabel the existing evidence. No model downloads
are attempted: the checkpoint is resolved from a pinned local cached snapshot,
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

NF4 uses the same local 8-GB RTX 5060 Laptop GPU and packages as Stage 3. BF16
uses unquantized weights, automatic device placement and the previously working
limits of GPU 6.5 GiB / CPU 6 GiB. Actual placement and dtypes are recorded in
each `provenance.json`. Actual package versions and Python version are frozen
per run. This is a reused environment, not a newly verified portable installation.

## Artifacts

- `results/stage4/nf4/`: 48 recipients, two positions, five modes (480 patches).
- `results/stage4/bf16/`: first two test scenes, eight recipients (80 patches).
- Each includes `freeze.json`, exact `source_snapshot/`, `environment.txt`,
  `provenance.json`, `prompts.json`, fresh `baseline.json`, `activations.npz`,
  fixed `axes.npz`, `interventions.json`, journals and integrity `manifest.json`.
- A completed `interventions.json` includes null checks, 16 one-token generation
  checks, per-patch delivered norms/projections and the complete descriptive summary.

The direction is estimated separately at each position from the original NF4
training rows and kept FIXED for BF16. BF16 is therefore a same-axis precision
check, not a refit of an optimal BF16 feature. Twelve-scene NF4 intervals are
exploratory scenario bootstrap intervals. Two-scene BF16 intervals are omitted.

No original Stage 3 evidence or manifest is rewritten. No Git commit, push or
permission change is part of this experiment.
