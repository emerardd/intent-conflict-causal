# Stage 5 reproduction

Run from the repository root on Windows PowerShell, inside this repository's
virtualenv (see the README for setup). The original frozen run was executed in a
pre-existing local environment; no other project's code or results were imported.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = (Resolve-Path -LiteralPath .\src).Path
$python = "python"   # this repository's activated virtualenv
```

## CPU-only tests and audits

```powershell
& $python -m pytest -q --basetemp .\.pytest_stage5_repro .\tests
& $python -m intent_conflict.stage5 --phase verify
& $python .\scripts\audit_stage5.py
```

The first command should report 37 tests or more if later tests are added. `verify`
checks the frozen source snapshot and exactly recomputes the saved summary. The
independent audit does not import the Stage 5 summarizer; it reconstructs identities,
cluster effects, bootstrap intervals, decisions, categorical flips, patch delivery,
and frozen hashes from raw artifacts.

## GPU rerun

The original output directory is immutable by convention and the runner refuses to
overwrite it. To rerun, first copy the repository to a clean location and change only
the config `output` field to a new, explicitly named directory before any model
response. Changing model, revision, data, layer, sites, thresholds, axes, or seed is a
new experiment rather than a reproduction.

```powershell
& $python -m intent_conflict.stage5 --phase run
```

Required local model snapshot:

- `Qwen/Qwen3.5-4B`
- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- NF4 load with bfloat16 compute

The recorded hardware was an RTX 5060 Laptop GPU with 8 GB VRAM. The frozen job used
48 baseline forwards, 576 intervention forwards, two null checks, and 24 one-token
generation checks; model-load-to-completion elapsed time was about 208 seconds. The
runner's frozen limit is 1,200 seconds. Different hardware or library kernels can
produce small numerical differences, especially because the saved logits show
bfloat16-sized steps.

## Artifact identities

`results/stage5/confirmation-v1/freeze.json` records the configuration, prompt IDs,
donor indices, source hashes, and Stage 4 axes reference hash before Stage 5 responses.
`source_snapshot/` contains the actual inference-time source. `environment.txt` is a
package inventory, not a portable lockfile. `provenance.json` records the resolved local
snapshot, model class, parameter dtypes, token IDs, and device map.

`baseline-progress.jsonl` and `interventions-progress.jsonl` are append-only journals.
The compact `baseline.json`, `activations.npz`, and `interventions.json` are the primary
raw analysis inputs. `independent_audit.json` is derived and can be regenerated.
