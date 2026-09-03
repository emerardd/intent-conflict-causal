# Stage 5b: equal-norm position by direction identity confirmation

Date frozen: 2026-08-29, before any model response on the Stage 5b scenarios.

## Motivation and narrow question

Stage 5 independently confirmed that paired full-state and local-direction replacement
at `answer` moved the A/B log-probability margin, while replacement at `pre_mapping`
was near zero. Its cross-applied direction components were not dose matched: the mean
natural cross projections were roughly 0.05--0.07 in norm, versus 0.58--0.96 for local
components. Consequently, the cross-direction null could reflect either direction
identity or insufficient dose.

Stage 5b asks only:

> At an exactly shared intervention norm, does the frozen answer-position direction
> have an effect at the answer position that exceeds random directions and the frozen
> pre-mapping direction, and is that direction contrast position dependent?

This is not a prompt-injection experiment, an awareness test, a tool-use experiment,
or a dissociation of authorization from the normative execute/block decision.

## Frozen model, representations, and new data

- Locally cached `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4 weights and bfloat16 compute.
- Decoder block index 16 only; no layer search.
- Target positions `pre_mapping` and `answer` only; no position search.
- Unit directions are the two NF4 directions already archived by Stage 4. They were
  fitted using Stage 3 training examples and are not refitted on Stage 5 or Stage 5b.
- Twelve new fictional scenario identities and tools, absent from Stages 1--5. Each
  has two authorization states by two A/B mappings, giving 48 prompts.
- The prompt grammar stays fixed. Candidate/distractor codewords are deterministic and
  unique by scenario.

## Frozen common scale

The common norm is derived without Stage 5 or Stage 5b responses:

1. Load the Stage 3 v2 formal activations and rows.
2. Restrict to Stage 3 `train` rows.
3. At each target position, pair rows with the same scenario, grammar, record order,
   and output mapping but opposite authorization.
4. For that position's frozen local unit direction, calculate the absolute paired
   donor projection for every training row.
5. Take the median separately at `pre_mapping` and `answer`.
6. Use the smaller of those two medians as the single shared scale `s`.

The derived value and both medians must be written to `freeze.json` before Stage 5b
model forward. Every fixed-axis intervention has exact requested norm `s`; there is no
alpha or scale sweep.

## Baseline gate and stopping

Record full-vocabulary first-token predictions on all 48 prompts. Each A/B mapping must
have at least 90% exact top-1 accuracy. If either fails, stop before intervention and
report the failure without revising prompts or selecting scenarios.

The run has a 1,200-second bound from model-load start. The runner refuses to overwrite
an existing output and retains append-only partial journals on failure.

## Intervention matrix

For recipient state `h`, frozen unit direction `d`, common scale `s`, and
`z=+1` when the paired donor is unauthorized and `z=-1` otherwise, write:

`h' = h + z s d`.

At each of the two target positions apply:

- the frozen `pre_mapping` direction;
- the frozen `answer` direction;
- ten fixed random unit directions, mutually orthogonal within the target-position
  bank and orthogonal to both frozen real directions.

Thus all 24 real/random fixed-axis conditions per prompt have exactly the same requested
norm. Random directions are generated from the frozen seed before model responses.

Secondary controls at each position are paired donor `full`, one natural-norm
`random_full`, and one null write. Full-state conditions are not equal-norm direction
tests; they only check whether the Stage 5 position effect appears again.

For every first-scenario prompt, run one-token greedy generation for `full`, both real
equal-norm directions, and the first random equal-norm direction at both positions.
Require agreement with the corresponding patched forward.

## Estimand and uncertainty

The per-intervention outcome is the change in `log p(B)-log p(A)` from recipient
baseline, signed so positive moves toward the paired donor's correct answer. Average
the four prompt effects within scenario. Random controls are first averaged over the
ten axes within each scenario. Report 10,000-replicate scenario bootstrap 95% intervals,
mapping-specific effects, exact top-1 flips, donor-directed flips, invalid outputs,
and delivered patch norms.

## Three primary predeclared checks

1. **Answer direction over random:** at `answer`, the answer direction minus the mean
   of ten equal-norm random directions has a bootstrap interval wholly above zero.
2. **Answer direction over equal-norm cross direction:** at `answer`, the answer
   direction minus the pre-mapping direction has an interval wholly above zero.
3. **Position by direction interaction:**

   `[answer-site(answer-axis - pre-axis)]
    - [pre-site(answer-axis - pre-axis)]`

   has an interval wholly above zero.

Each is reported separately. Passing all three supports equal-dose position-specific
direction identity. Failure is retained as a bounded negative result. Secondary results
include each raw effect, both mappings, the answer-axis position contrast, the local-axis
position contrast, and full-state answer-minus-pre; none may replace a failed primary.

## Claim boundary and delivery

Even if all checks pass, the result is a causal log-margin effect in a controlled
forced-choice task. It does not show natural behavioral control, pure authorization,
prompt-injection awareness, or conscious knowledge of violating user intent.

Retain config and source snapshots, reference hashes, derived scale, prompts and token
positions, baseline/intervention journals, raw probabilities, axes, patch delivery,
environment, manifest, exact verification, independent arithmetic recomputation, tests,
run log, strongest counterevidence, and a complete report. No commit or push is included.
