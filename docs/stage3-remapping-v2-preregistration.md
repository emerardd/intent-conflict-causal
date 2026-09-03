# Stage 3 remapping-only v2: bounded follow-up after failed factorial pilot

Frozen on 2026-08-28 AFTER seeing the 32-row factorial pilot and BEFORE seeing
any v2 response. The v1 gate remains failed, and its formal/causal stages are not
run. This is a transparently adaptive narrowing, not an independent replication
of the original scientific question. Only this one additional prompt protocol
will be tried in this work session; if its pilot fails, stop without more prompt
tuning or model switching.

The v1 model generated B in all 24 noncanonical policy/mapping rows. Therefore v2
removes the reversed controller rule entirely and asks only whether the candidate
is authorized, with a trial-specific A/B response legend. It retains both the
permission and prohibition and the fixed candidate JSON. The system has no A/B
mapping. Policy is always normal. **Authorization and execute/block semantics are
again coupled**; success can rule out literal output-token identity, NOT establish
authorization independently of semantic action choice or natural violation.

Two new pilot identities and codes (toy bridge, paper mobile) replace the v1
pilots. Pilot size is 16 (2 scenarios x 2 orders x 2 authorization x 2 mappings).
The untouched 24 formal scenes and 8/4/12 split from v1 remain unused until this
pilot passes. Formal size is 288: 64 train, 32 validation, 192 test (half test uses
the never-trained permission grammar). No formal response has yet been seen.

Two markers precede the legend with no controller rule between them. The retained
internal array name `pre_policy` now means the end of case facts, not a manipulated
policy. `pre_mapping` remains the primary position, and `answer` is descriptive.
Matched mappings have identical earlier prefixes; input lengths and numerical
prefix invariance are audited. The v1 post-hoc diagnostic showed prefix activation
length sensitivity (not content sensitivity in the equal-length control), so
suffix-length mismatches must be disclosed and cannot be called anticipation.

All behavioral/generation thresholds remain 0.90. Each pilot mapping has 8 rows;
all 8 must be correct to pass. Real greedy generation is checked on all pilot
rows and the first scenario of each formal split. Any failed screen stops all
formal work. Wrong outputs are never removed from primary metrics.

Representation: train-only unauthorized-minus-authorized contrast; remove only
the train expected-token mean-difference span (not the confounded action contrast).
Select the earliest primary-position layer maximizing minimum validation AUROC
across mapping subgroups and both cross-mapping transfers. Require >0.75 validation
and >0.75 in each seen-grammar test mapping. Test unseen grammar is separate.
Report all sites and the original Stage 2 layer-16 direction on every mapping.
Cluster uncertainty by scenario, not prompt.

If both behavioral and representation gates pass, apply the same frozen +/-0.5,
+/-1 and +/-2 train-gap interventions to all 12 test scenarios, seen grammar,
candidate-first, all four authorization/mapping cells (48 recipients). Compare
with the expected-token direction (if nonzero) and two antipodal random orthogonal
axes at +/-1. No action-direction control: it is confounded with authorization.
The primary outcome is the map-aligned half plus-minus margin at alpha=1. Require
positive scenario-bootstrap lower CI, positive mapping-cell means, and positive
main-minus-random lower CIs to label it a mapping-consistent causal effect. This
still does not identify a uniquely authorization-semantic mechanism. The first
test scenario gets real one-token greedy generation under +/-1 intervention.

Source/config snapshots for this v2 will include all implementation edits since
v1. All v1 files remain preserved; no previous gate or result is overwritten.
