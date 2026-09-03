# Stage 5b reproduction

Run from the repository root on Windows PowerShell, inside this repository's
virtualenv (see the README for setup). The original frozen run was executed in a
pre-existing local environment; no other project's code or result was an
experimental input.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
$python = "python"   # this repository's activated virtualenv
```

## CPU-only verification

```powershell
& $python -m pytest -q --basetemp .\.pytest_stage5b_repro .\tests
& $python -m intent_conflict.stage5b --phase verify
& $python .\scripts\audit_stage5b.py
```

The completed delivery has 41 tests before post-run delivery additions. `verify` checks
the inference-time source snapshot, exactly recomputes the stored analysis, and checks
equal-norm delivery. `audit_stage5b.py` deliberately does not import the Stage 5b
analysis code; it independently derives the Stage 3 training scale, reconstructs row
identities and donor pairs, recomputes the three primary clustered bootstraps, and
checks categorical outcomes, random axes, patch norms, generation, nulls, and hashes.

## GPU rerun

The runner refuses to overwrite `results/stage5b/confirmation-v1`. For a genuine rerun,
copy the repository to a clean location and change only the config output to a new
explicit directory before any model response. A change to data, seed, scale method,
axis bank, model, revision, precision, layer, position, gate, or statistic is a new
experiment.

```powershell
& $python -m intent_conflict.stage5b --phase run
```

Required cached model:

- `Qwen/Qwen3.5-4B`
- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- NF4 weights with bfloat16 compute

Recorded hardware was an 8 GB RTX 5060 Laptop GPU. The run used 48 baselines, 1,344
patched forwards, two null checks, and 32 one-token generation checks. Model-load start
to completion was about 407.8 seconds, below the frozen 1,200-second bound. Different
kernels can move bfloat16-scale margins and delivered norms slightly.

## Artifact map

- `freeze.json`: config, common-scale derivation, reference/source hashes, identities,
  donor indices, and axis numerical checks, written before Stage 5b responses.
- `frozen_axes.npz`: two real directions and twenty position-specific random controls.
- `prompts.json`: exact token IDs and measurement positions.
- `baseline-progress.jsonl`, `interventions-progress.jsonl`: append-only raw journals.
- `baseline.json`, `activations.npz`, `interventions.json`: primary raw artifacts.
- `provenance.json`, `environment.txt`: model snapshot/runtime metadata.
- `independent_audit.json`: regenerable independent arithmetic audit.
- `manifest.json`: final artifact hashes.

`environment.txt` is an inventory, not a portable lockfile. A clean-machine model rerun
has not been performed.
