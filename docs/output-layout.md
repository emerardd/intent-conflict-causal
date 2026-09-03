# Output layout

Runs write to `results/` and `plots/`. **`results/` is tracked**: every raw
artifact ships with the repository so the audit scripts can be run on a fresh
clone without a GPU. `plots/` is not tracked — it is regenerable, and the
figures the reports embed are checked in under [`figures/`](figures/).

This document describes what a completed run contains, so the paths referenced
throughout the stage reports are legible without opening every file.

## `results/`

Stage 1 and Stage 2 (CLI runs) write flat, per-run files named after the config:

```
results/
  qwen_formal.json                  Full run record: config, behavior, per-layer
                                    scores, selected layer, intervention effects
  qwen_formal_activations.npz       Archived residual-stream activations
  qwen_formal_robustness.json       Same-authorization wording-swap controls
  stage2_comparison.json            Cross-run synthesis over the Stage 2 runs
```

Stages 3–5b write one self-contained directory per run, under a stage root:

```
results/
  stage3/
    audit.json                      Saved output of scripts/audit_stage3.py
    delivery.json                   Delivery status and current-source hashes
    manifest.json                   Artifact manifest with per-file hashes
    screen-v1/                      v1 pilot — failed the screening gate
    screen-v1-preflight-error/      Preflight failure, kept as a record
    remapping-v2/
      screen-v1/                    v2 screening run
      formal-v1/                    v2 formal run (the reported one)
  stage4/
    bf16/  nf4/                     One directory per precision
    matched_precision_comparison.json
    audit.json  delivery.json  manifest.json
  stage5/confirmation-v1/
  stage5b/confirmation-v1/
```

### Inside a stage run directory

| File | Contents |
| --- | --- |
| `prompts.json` | Every rendered prompt, with its scenario, condition and mapping cell. |
| `baseline.json` | Un-intervened forward passes: decision logits, margins, generations. |
| `freeze.json` | The frozen analysis decisions (layer, positions, direction, scale) committed before intervention. |
| `interventions.json` | One record per intervention: donor pair, axis identity, patch norm, resulting logits. |
| `activations.npz` | Archived residual-stream activations for the measured sites. |
| `frozen_axes.npz` / `axes.npz` / `intervention_axes.npz` | The real and random directions used, frozen before the run. |
| `analysis.json` | Computed effects, bootstrap intervals and pre-declared checks. |
| `independent_audit.json` | Result of the standalone audit that does not import the analysis code. |
| `manifest.json` | Per-file hashes for every artifact in the run. |
| `provenance.json` | Model name, resolved revision, precision, seeds, library versions. |
| `environment.txt` | Captured runtime environment. |
| `*-progress.jsonl` | Append-only progress log, written during long runs so an interrupted run is diagnosable. |
| `source_snapshot/` | Copy of the exact config, pre-registration and source files as of run time. |

The `source_snapshot/` directory is why an audit is meaningful: it records the
code that produced the artifacts, so a later change to `src/` cannot silently
alter the interpretation of a completed run.

Runners refuse to overwrite an existing confirmation directory. A genuine rerun
means copying the repository to a clean location and pointing the config at a
new explicit output directory — and any change to data, seed, scale method,
axis bank, model, revision, precision, layer, position, gate or statistic makes
it a new experiment rather than a reproduction.

## `plots/`

One directory per run, named by the config's `plot_dir`:

```
plots/
  qwen_formal/
    behavior_margin.svg   layer_auroc.svg   patching_effect.svg
    formal-report.md      Auto-generated numeric summary
  stage2_comparison/
    layer_auroc_comparison.png   causal_controls_comparison.png
    qwen_precision_agreement.png
```

The figures that the stage reports embed are checked in separately under
[`figures/`](figures/), so the reports render correctly in a fresh clone.
