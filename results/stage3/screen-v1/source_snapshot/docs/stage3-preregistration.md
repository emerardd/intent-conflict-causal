# Stage 3 v1: authorization, controller action, and token mapping

Frozen locally on 2026-08-28, before any Stage 3 model response is observed.
The runner saves UTC timestamps, source/config SHA-256, the complete dataset and
prompt-token hashes before inference. This is a local preregistration, not an
externally timestamped registration. Stage 1/2 artifacts will not be overwritten.

## Question and limits

Does an authorization contrast transfer across both output-token remapping and
the **requested controller action**? Does a feature-isolated intervention move
the decision in the direction predicted by the trial-specific controller rule
and token mapping?

The task is an inert simulation. NORMAL executes authorized and blocks
unauthorized proposals; REVERSED does the opposite. These are experimentally
assigned controller rules, NOT spontaneous violations of a real user's intent.
The assigned semantic action and observed model action are stored separately.
Success cannot establish consciousness, natural self-monitoring, or knowledge
of violating a real user. Even a mapping-invariant signal may encode the ordinary
normative execute/block recommendation; the narrower claim is independence from
this trial's requested action and letter output, not a uniquely semantic concept.

## Dataset, splits, and factorial balance

Two pilot scenarios are never used for train/validation/test. There are 24 new
formal scenario identities: 8 train, 4 validation, 12 test. Every scenario uses
two globally unique six-letter inert object codes, unrelated to Stage 2 codes.
Each scenario has authorization x controller policy x A/B mapping (2x2x2), with
candidate-first and candidate-last constraint ordering. The training permission
grammar is shared by train/validation/test; a second, never-trained grammar is
evaluated only on test. This gives 32 pilot and 576 formal prompts.

The record contains both a permission and a prohibition, with their binding
swapped while candidate command and object strings are fixed. Output mapping is
absent from the system prompt. No Stage 2 test scenarios are reused. These remain
hand-constructed synthetic tasks; new grammar means only one held-out grammar,
not general natural-language generalization.

## Measurements

Read decoder-block outputs at three positions: the final fully preceding token
at FACTS_END (before policy), RULE_END (before mapping), and the final chat-prefix
token (before answer). Exact tokenizer offset alignment and token-prefix equality
are verified. The last decoder layer is excluded to avoid final-norm ambiguity.
Identical causal prefixes should have approximately identical earlier states;
numerical invariance is audited, not assumed to be bit-exact across lengths.

Record A/B log probabilities, log p(B)-log p(A), global-vocabulary argmax/token,
top-five tokens, A+B mass, assigned action, inferred observed action, and exact
prompt IDs. Actual greedy generation (up to four tokens) is audited for every
pilot row and the first scenario of each formal split (all its cells/grammars).
Generation must produce exactly the expected letter after stripping whitespace;
first token must equal the single-forward global argmax. No constrained decoding.

## Screen / stopping rules

Use one Qwen3.5-4B checkpoint at NF4, the same local environment as Stage 2.
Pilot must reach >=0.90 global-top1 accuracy in every policy x mapping cell and
>=0.90 exact generation accuracy. Otherwise stop: do not run formal inference or
silently revise the prompt. Any later revision requires a new version and pilot.

Formal behavior checks use train and validation for selection eligibility; each
policy x mapping cell must reach >=0.90 global-top1 accuracy. Test behavior and
generation are reported independently; a failed test gate forbids causal
authorization claims and cancels the planned intervention phase. Wrong outputs
are retained. No post-hoc behavior filtering for primary analyses.

## Representation analysis and frozen selection

Mean differences use training scenarios only. Positive authorization means
unauthorized; positive action means assigned BLOCK; positive token means expected
B. Factorial balance separates these three labels. Cross-action fits train on
assigned EXECUTE and test on assigned BLOCK, and vice versa; cross-mapping and
cross-policy fits are also reported. AUROC is directional (never max(AUC,1-AUC)).

At each site, remove the span of train-only assigned-action/token mean differences
from the authorization contrast, using SVD; numerically zero nuisance contrasts
are dropped. Normalize the residual direction, recording its retained norm.
Site score is the minimum validation AUROC across the four policy x mapping cells
and the two cross-action transfers. The primary site is **pre_mapping**; select
the earliest layer maximizing this validation-only score. Other positions and
layers are descriptive, not alternative primary sites after a failure.

Continue to intervention only if the primary validation score >0.75 and the
selected train-only direction has test AUROC >0.75 in each seen-grammar cell and
both assigned-action subgroups. The unseen-grammar score is reported separately;
it is not used to choose layers or reinterpret a failed primary result.

Cluster all uncertainty by scenario, not prompt. Report scenario bootstrap CI
and a one-sided sign test on held-out scenario paired authorization projections.
The p value tests alignment, not the semantic identity of the feature. Stored
Stage 2 layer-16 train direction transfer is secondary, with all mapping/policy
cell AUROCs reported (no selection on them).

## Feature-isolated intervention

Use all 12 test scenarios in the seen permission grammar, candidate-first order,
all eight factorial cells: 96 recipient prompts. Primary direction is the
selected residualized train authorization direction. Unit displacement is the
train unauthorized-minus-authorized projection gap (not tuned on test).
Test +/-0.5, +/-1 and +/-2 units. Controls at +/-1 unit: assigned-action direction,
expected-letter direction if nonzero, and two fixed-seed random directions
orthogonal to authorization. Each control uses the SAME axis with opposite signs.

Primary causal statistic at alpha=1 is half the plus-minus B/A-margin difference,
aligned to the expected effect of more unauthorized evidence under each trial's
policy and mapping. Bootstrap scenario means (5000 samples). Report per-cell
effects, absolute/relative magnitude, global-top1/format/forced-choice flips,
and the full dose response. No claim of semantic specificity from random
controls alone. Require positive lower 95% CI, positive means in every cell,
and positive lower CI for main-minus-each-random paired differences to call
the intervention a conditional rule-consistent effect. Report all failures.
Actual one-token greedy generation under +/-1 intervention is checked on the
first test scenario. Baseline null patches must reproduce logits within 1e-4.

## Delivery

JSON rows, prompt IDs, source snapshots, selected-site activations/directions,
hash manifest, exact dependency snapshot and a read-only recomputation command.
All-layer activations stay local; selected-site data suffice for primary metrics.
No model downloads, base-model training, real external actions, Git commits or
GitHub pushes are part of this run. Natural injection/tool-use transport is a
future project gated on the controlled results, not an implicit claim of Stage 3.
