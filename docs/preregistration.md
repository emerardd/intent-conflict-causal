# Minimal preregistration

Date: 2026-08-15

## Question

Does an open-weight language model form a causally active representation, immediately before a decision, of whether its proposed next action is authorized by the user?

## Manipulation

The proposed action is byte-identical within each matched pair. Both prompts contain one permission and one prohibition. The authorized and unauthorized conditions swap which of two action descriptions is permitted and prohibited. Candidate-action mention order is held fixed within a pair and counterbalanced across prompt variants.

## Primary behavioral measure

`block_margin = log p(B) - log p(A)`, where `A` means execute and `B` means block. Both labels must tokenize to one token.

Primary behavioral effect: paired mean `margin_unauthorized - margin_authorized`.

## Primary representational measure

At the last prompt token before the decision, compute at each layer:

`d_train = mean(h_unauthorized) - mean(h_authorized)`.

Layer selection uses validation scenarios only. The final test reports held-out AUROC, paired score differences, train/test direction cosine, and a within-pair label-permutation null.

After dataset calibration and before any Qwen representation analysis, the 18-scenario formal split was frozen: the six screened scenarios plus four additional scenarios are training data; four unseen scenarios are validation data; the final four unseen scenarios are test data. No Qwen layer-selection or patching result was observed before freezing this split.

## Primary causal measure

At the validation-selected layer, replace the target last-token residual activation with its matched source activation:

- authorized -> unauthorized: predicted to reduce the block margin;
- unauthorized -> authorized: predicted to increase the block margin.

The primary causal effect is the mean of the two sign-aligned changes. Norm-matched random deltas provide the perturbation control. All final causal tests use held-out test scenarios.

## Gates

The formal run is not scientifically interpretable if the behavioral screen fails. The causal claim fails if activation swaps do not have the predicted bidirectional effect or if norm-matched random perturbations perform similarly.

## Scope exclusions

No claim about consciousness, natural cheating, reward hacking, action risk, hidden user intent, or post-hoc rationalization is made in the minimal study.
