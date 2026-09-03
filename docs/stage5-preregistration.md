# Stage 5: position by direction identity confirmation

Date frozen: 2026-08-29, before any model response on the new Stage 5 scenarios.

## Question and claim boundary

Stage 4 found, on already inspected Stage 3 test scenarios, that a paired residual
replacement at `pre_mapping` had near-zero effect while `answer` replacement moved
the A/B margin toward the paired donor. At `answer`, the locally fitted one-dimensional
component was substantially larger than its norm-matched random control. Stage 4 was
exploratory. This confirmation asks:

> On entirely new scenarios, is the causal response primarily tied to the token
> position, to the identity of the fitted direction, or to both?

This is not a test of natural prompt injection, actual tool use, conscious awareness,
or authorization independent of the normative execute/block decision. A positive
result identifies a position- and/or direction-specific causal effect on a controlled
A/B log-probability margin, not a complete mechanism or a behavioral safety control.

## Frozen model, data, and sites

- Model: locally cached `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4 weights with bfloat16 compute.
- Layer: decoder block index 16. No layer search.
- Target positions: `pre_mapping` and `answer`. No position search.
- Source directions: the two NF4 directions already saved by Stage 4 for those
  positions. They were fitted from the 64 Stage 3 training examples only. They are
  not refitted on Stage 5.
- Data: 12 new fictional scenario identities and tools, four prompts per scenario
  (two authorization states by two A/B mappings), 48 prompts total. None of these
  identities occur in Stages 1--4. The prompt grammar and task remain fixed so that
  this is a confirmation rather than a task-transfer experiment.
- Candidate/distractor codewords are deterministically generated from the frozen
  seed and are unique across scenarios.

## Baseline gate and stopping rule

Before intervention, record unconstrained full-vocabulary top-1 predictions for all
48 prompts. Each mapping must achieve at least 90% exact first-token accuracy. If the
gate fails, stop before intervention and report the behavior failure. Do not revise
the prompt or select a successful scenario subset.

The NF4 job has a 20-minute runtime bound after model load begins. Partial journals
are retained on failure. No replacement output directory may be silently overwritten.

## Paired interventions

For each recipient, the donor has the same scenario, candidate, prompt grammar, and
output mapping, but the opposite authorization state. At each target position let
`delta = h_donor - h_recipient`. For each frozen source direction `d`, apply:

1. `parallel`: `h + (delta dot d)d`;
2. `random_parallel`: a fixed random direction, orthogonal to both frozen source
   directions, with the same signed coefficient and therefore the same norm;
3. `full`: `h + delta` (once per target position);
4. `random_full`: the same displacement norm on the fixed random direction, with
   sign determined by the donor authorization state;
5. `null`: write the unmodified recipient vector for the first recipient at each
   target position.

This is the frozen 2 target positions by 2 source-direction identities design. There
is no alpha sweep, dose-response selection, prompt filtering, or post-hoc layer search.
For every non-random `full` and `parallel` intervention in the first scenario, run an
actual one-token greedy generation and require agreement with the patched forward.

## Estimand and uncertainty

For each intervention, compute the change in `log p(B)-log p(A)` relative to the
recipient baseline and multiply by the sign of the donor's correct answer. Positive
values move toward the opposite-authorization donor's answer. Average the four prompt
effects within each scenario, then estimate means and paired differences across the
12 scenario clusters. Report deterministic 10,000-replicate cluster bootstrap 95%
intervals, both mapping-specific means, global top-1 flips, donor-directed flips,
invalid first tokens, and natural donor-gap recovery. The margin is primary; categorical
flips are secondary and are not required because all baselines may be far from the
decision boundary.

## Four predeclared decision checks

1. **Answer local-direction specificity:** `answer/answer-axis parallel` minus its
   norm-matched random control has a bootstrap interval wholly above zero.
2. **Answer full-state positive control:** `answer/full` minus `answer/random_full`
   has an interval wholly above zero.
3. **Position interaction:** the local-direction effect at `answer` minus the local-
   direction effect at `pre_mapping` has an interval wholly above zero.
4. **Bounded pre-mapping full effect:** the entire interval for the unadjusted
   `pre_mapping/full` mean lies inside `[-0.1, 0.1]` log-margin units.

Checks 1--3 are positive-effect confirmations. Check 4 is an equivalence-style bound
chosen before responses and is not the same as failing to reject zero. Each check is
reported separately; there is no rule that all four must pass for the experiment to
be publishable as a bounded result. Cross-applied directions are diagnostic secondary
comparisons used to distinguish position from direction identity.

## Delivery

Retain the prompt/token identities, frozen directions and reference hashes, baseline
and intervention journals, raw probabilities, delivered patch norms/projections,
environment, source snapshot, result manifest, independent arithmetic recomputation,
tests, run log, and a report that includes the strongest counterevidence and all claim
limits. No Git commit or push is part of the experiment.
