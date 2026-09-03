# Stage 3 reproduction and evidence map

This project distinguishes **recomputing stored evidence** from **rerunning the
model**. Passing an artifact audit does not independently reproduce GPU inference
or prove the semantic interpretation of a latent direction.

## Read-only CPU audit

From this repository's root, with Python >=3.11 and NumPy available:

```powershell
python scripts/audit_stage3.py
```

The audit independently computes behavior, a pairwise-comparison AUROC without
scikit-learn, all 21 site scores, validation-only layer selection, train-only
direction and scenario-level projection uncertainty. If intervention output
exists, it recomputes paired effects, scenario bootstrap intervals, mapping-cell
means, categorical changes and the frozen support rule. It imports no model
loader and needs no GPU, model weights, or Stage 2 archive.

Pilot behavior alone needs only Python >=3.11, without even NumPy:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
python -m intent_conflict.stage3_audit --run results/stage3/screen-v1
python -m intent_conflict.stage3_audit --run results/stage3/remapping-v2/screen-v1
```

Adding `--include-activations` checks archive shapes, row identity and finiteness
and requires NumPy. Exit code zero means the stored evidence is internally
consistent; `screen_passed: false` is a valid experimental result, not an audit
failure. The v1 pilot is expected to have that result.

## Model-inference environment

After local finalization, the CPU audit command also verifies the artifact
manifest and current delivery source hashes. `results/stage3/audit.json` is the
saved audit, `delivery.json` records status/current-source hashes, and
`manifest.json` records raw artifact hashes. Intentional later source edits will
correctly fail the current-source integrity check until a new delivery is
explicitly exported. Do not rewrite original freezes to hide a change.

The actual runs used an RTX 5060 Laptop GPU (8 GB), an already cached
`Qwen/Qwen3.5-4B` checkpoint, NF4 weights and bfloat16 computation. The observed
checkpoint revision was `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Native chat formatting is used with thinking disabled by the existing tokenizer
helper. No constrained A/B decoder is used. Generation is greedy, `use_cache=False`,
at most four new tokens for baseline checks and one for patched checks.
The Python runtime was 3.12.10.

Important observed packages:

| Package | Observed version |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.15.0 |
| bitsandbytes | 0.50.0 |
| accelerate | 1.14.0 |
| numpy | 2.3.5 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |

Each run's `environment.txt` is the full observed package inventory, **not a
portable lockfile**. It includes unrelated packages and a sibling editable
project. A clean-machine installation was not tested. The existing local
environment was reused; no model was trained or downloaded in Stage 3.
The current loader records the resolved revision but does not pin it in config;
a reproduction must supply the same cached snapshot explicitly in its own
versioned configuration to avoid silently using another cached revision.

## Actual inference commands

These commands describe the runs; completed output directories are intentionally
write-protected by the runner's no-overwrite check. Do not delete evidence to
rerun. Use a separate copy of the project without the output directories, or a
new, explicitly versioned configuration/output root. Do not edit the original
frozen config or preregistration.

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

# v1: pilot only; formal is prohibited by its failed gate.
python -u -m intent_conflict.stage3 --phase screen --config configs/stage3_factorial.json

# v2: pilot, then formal only after its pilot passes.
python -u -m intent_conflict.stage3_remap --phase screen
python -u -m intent_conflict.stage3_remap --phase formal

# Conditional: only if formal behavior and representation gates both pass.
python -u -m intent_conflict.stage3_remap --phase intervene
```

The formal analysis also reads the existing local Stage 2 NF4 result and its
activation archive for a secondary direction-transfer comparison. These old
archives were not part of the prior GitHub export; a clean checkout without them
cannot run that secondary analysis unchanged. The independent Stage 3 CPU audit
does not have this dependency.

## Artifact layout

- `results/stage3/screen-v1-preflight-error/`: preserved tokenizer-preflight
  failure; no model forward, no behavioral result.
- `results/stage3/screen-v1/`: completed failed factorial pilot and post-hoc
  prefix diagnostic.
- `results/stage3/remapping-v2/screen-v1/`: completed remapping pilot. The outer
  directory is protocol version v2; inner `v1` means that protocol's first run.
- `results/stage3/remapping-v2/formal-v1/`: formal data, if its pilot gate passed.

Within each completed run, `freeze.json` and `source_snapshot/` record the exact
pre-inference code/config/data; `prompts.json` stores messages, token IDs and
measurement positions; `provenance.json` records the actual checkpoint and token
IDs; `baseline.json` stores per-example outputs; `activations.npz` stores all
recorded sites. Formal runs additionally export `analysis.json` and
`selected_site.npz`. Conditional interventions have a separate freeze, axes,
per-forward outputs and generation audit. A hash manifest detects later changes;
it is a local integrity check, not an externally timestamped preregistration.

All artifacts remain local unless separately uploaded. Model weights are never
included. Preserve the failed version when sharing the successful follow-up.
