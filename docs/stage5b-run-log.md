# Stage 5b run log

## 2026-08-29 scope correction and design

- Continued the latest `intent-conflict-causal` project; did not switch to the older
  prompt-injection research question. The adjacent project supplied only the existing
  Python environment.
- Restricted Stage 5b to the one unresolved Stage 5 confound: cross-direction natural
  projection norms differed by an order of magnitude.
- Fixed one model revision, NF4 precision, layer 16, two target positions, two archived
  directions, ten random axes per target, 12 new scenarios, one train-derived common
  norm, three primary comparisons, mapping gate, and runtime bound.
- A first read-only status command had a PowerShell parse error from piping directly
  after `foreach`; it performed no model call and wrote no experiment file. Replaced it
  with an explicit collection and confirmed all reference artifacts existed.

## Preflight

- Added config, preregistration, new data generator, runner, and four Stage 5b tests.
- Static compile passed.
- Repository test target: 41/41 passed before any Stage 5b model response.
- Independently printed the frozen scale formula output before responses:
  `pre_mapping` median 0.59587675, `answer` median 1.14314766; common scale 0.59587675.
- `git diff --check` passed except the existing README Windows line-ending warning.

## Frozen run

- Output: `results/stage5b/confirmation-v1`; overwrite refused by code.
- Source snapshot, axes, scale derivation, prompt IDs, donor identities, environment, and
  provenance were stored before the first Stage 5b forward.
- Baselines: 48/48 completed; normal and reversed mapping each 24/24 correct; gate passed.
- Interventions: 1,344/1,344 completed with no condition removal or parameter change.
- Nulls: 2. Generation audits: 32, all matching corresponding patched forwards.
- Elapsed from model-load start: 407.812 seconds; frozen 1,200-second bound not approached.
- All three primary checks passed in the frozen analysis.

## Post-run audit and interpretation

- Frozen source verification, exact summary recomputation, and requested equal-norm
  delivery check passed.
- Added `scripts/audit_stage5b.py` after results. It does not import Stage 5b analysis
  functions and independently recomputes the scale, identities, random banks, primary
  statistics, decisions, categorical outcomes, delivery, and hashes. Audit passed.
- Inspected all ten random-axis means. At the answer position they ranged from −0.05469
  to +0.01823, versus 0.1875 for the answer direction.
- Inspected all categorical outcomes: no top-1 changes among 1,344 interventions.
- Wrote report and reproduction instructions after result inspection; neither is claimed
  to be frozen inference-time source.

No model training, commit, push, release, or remote permission change occurred.
