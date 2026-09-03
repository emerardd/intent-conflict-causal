# Stage 5 run log

## 2026-08-29 design and preflight

- Preserved all Stage 1–4 files and raw outputs.
- Added a new Stage 5 config, preregistration, 12-scenario generator, runner, and tests.
- Froze one model revision, NF4 precision, decoder layer 16, two positions, two Stage 4
  directions, 12 new scenarios, mapping gate, four decision checks, and a 20-minute
  runtime bound.
- Initial full test run: 36 passed, one new unit test failed because its synthetic
  fixture supplied only one of the 12 conditions required by the complete summarizer.
  No model forward had run. Fixed the fixture to construct all predeclared conditions;
  no prompt, threshold, model, axis, site, or statistic changed.
- Second preflight: 37/37 tests passed; `git diff --check` passed with only an existing
  README line-ending warning.

## 2026-08-29 frozen NF4 run

- Output: `results/stage5/confirmation-v1/`; runner would refuse overwrite.
- Source snapshot and config written before any Stage 5 forward.
- Baselines completed: 48/48 forwards. Mapping accuracy normal 24/24; reversed 23/24.
  Gate passed because both exceeded the frozen 90% threshold.
- Interventions completed: 576/576 forwards; two null patches; 24 generation checks.
- Elapsed from pinned model load start: 208.297 seconds; runtime bound not approached.
- All 24 patched generations agreed with patched forward top-1.
- Four predeclared checks passed according to the frozen analysis.

## Post-run checks

- Exact summary recomputation and frozen source-snapshot verification passed.
- Added a post-run independent audit that does not call the frozen Stage 5 summarizer.
- Independent audit passed identity, arithmetic, bootstrap, decision, categorical,
  generation, null, patch-norm, and hash checks.
- One combined final-delivery command accidentally omitted the explicit `tests` target.
  Pytest recursively collected the parent workspace and immutable source snapshots, then
  stopped on duplicate-module import mismatches. This was a command-scope error, not a
  project-test failure; no result artifact changed. The flow was stopped and replaced
  with the previously validated command explicitly targeting this repository's `tests/`.
- Inspected all top-1 changes: seven occurred, all on the single baseline A/B tie and
  none toward the paired donor answer.
- Computed post-run descriptive intervention norms to calibrate the cross-axis claim;
  these do not alter any frozen decision.
- Wrote the report and reproduction instructions after inspecting results; neither is
  represented as pre-response source.

No model was trained. No Git commit, push, release, or remote permission change occurred.
