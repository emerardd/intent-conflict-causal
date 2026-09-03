# Stage 3 run log

## 2026-08-28, pre-inference tokenizer compatibility correction

The first pilot attempt stopped before any model forward pass: the tokenizer
identity check treated Transformers 5.15's native `BatchEncoding` as a list.
Read-only diagnosis confirmed the rendered-text and native chat paths both
produce exactly the same 214 token IDs for the first pilot prompt. The check was
fixed to use the existing `apply_chat_template` normalizer and a regression test
was added. No prompt, split, metric, threshold, or scientific hypothesis changed.
The failed attempt's freeze and source snapshot are preserved in
`results/stage3/screen-v1-preflight-error`. This was a code preflight failure, not
a failed behavioral gate and not a completed experimental run.

## 2026-08-28, completed factorial v1 pilot

The unchanged scientific protocol was rerun after the tokenizer-only correction.
All 32 prompts completed, including unconstrained greedy generation. Correct
global-top1/generation outputs: 20/32. The normal-rule/normal-map cell was 8/8;
each of the other three cells was 4/8. All 24 noncanonical rows generated B.
All outputs obeyed the A/B format and generation agreed with forward argmax.
The frozen >=0.90-per-cell gate failed. No v1 formal or intervention run occurred.
The independent arithmetic truth-table audit reproduced these results and
verified snapshots, identities, and the (32,3,7,2560) activation archive.

The pre-policy identical-prefix audit found nonzero differences across prompts
of different total lengths. A clearly post-hoc four-forward diagnostic, saved
as `prefix_diagnostic.json`, reproduced the baseline exactly. An equal-length
suffix-content change and a prefix-only run both gave exactly identical recorded
prefix states; the 214-vs-216-token condition gave max absolute difference 0.0625
and relative L2 difference 0.0131. This is evidence of length-sensitive numerical
or implementation behavior, NOT evidence that the model read future content.
The diagnostic did not repair the failed behavior gate or generate formal data.

## 2026-08-28, prospectively frozen bounded remapping v2

After seeing v1, a separate local protocol was frozen: remove controller-rule
reversal, simplify instructions to authorization auditing, and retain only A/B
mapping reversal. This is an adaptive task redesign, not a one-variable causal
diagnosis of why v1 failed. Two new pilot scenarios and codes were used. The
24 formal scene identities remained untouched. Thresholds stayed at 0.90. Only
this one extra prompt protocol was allowed in the work session.

The v2 pilot completed 16/16 correct global-top1 and exact greedy outputs, 8/8
in each mapping. Prefix states matched exactly across mappings at both earlier
positions. Independent artifact audit passed before launching formal inference.
The v2 source/config/preregistration snapshots are separate from v1 snapshots;
the original failed experiment has not been overwritten or relabeled a success.

## 2026-08-28, completed formal v2 and frozen interventions

All 288 formal global-top1 outputs were correct. All 32 prospectively selected
greedy-generation checks were correct and agreed with forward argmax. The
validation-only primary selection chose layer index 16, before mapping, with
validation and seen-grammar test AUROC 1.0; unseen-grammar test AUROC was 0.897569
in both mappings. All five eligibility gates passed. The independent audit
recomputed behavior and all 21 sites before interventions started.

All 48 recipients completed the frozen direction/strength grid: 288 main-axis
and 192 random-axis forwards, 480 total. The token direction at the pre-mapping
site was zero, so the preregistered nonzero-token-direction control did not apply.
The auth/action confound rules out a separate action control in v2.

The primary aligned mean was -0.0065104, 95% scenario-bootstrap CI
[-0.0195312, 0.0065104]. Both mapping-cell means were negative. Neither paired
main-minus-random interval excluded zero. All 480 outputs retained A/B format and
none changed its global-top1 or within-A/B choice. Eight patched generation
checks agreed with their forward results. The causal-support gate failed.
No alternate layer, site, strength or subset was tried to rescue the result.

The frozen analysis was exactly recomputed. A separate NumPy/stdlib audit
recomputed all 21 AUROC sites without sklearn, selection, behavior, projection
uncertainty, intervention effects, flips, random-control comparisons and gates.

## 2026-08-28, post-hoc delivery diagnostic and final local checks

After the negative intervention result, exactly four additional forwards checked
the first frozen recipient: baseline, null, and the already-tested +/-1 patches.
An observer hook registered after the patch saw nonzero delivered perturbations:
requested norm 0.610585, actual minus/plus norms 0.612268/0.612308; projections
-0.612042/+0.612082. Baseline hidden norm was 10.610829. All four margins and
argmax outputs exactly reproduced the saved corresponding results. Both hidden
states and output logits had bfloat16 dtype. The patch did not disappear in dtype
casting, but this check is NOT a behavior-changing positive control.

All 30 tests passed, including five post-run independent-audit and stopping-guard
regressions. `git diff --check` passed (only Git's LF-to-CRLF informational warning).
No model was trained or downloaded, and no commit or push was performed.
Final source/config/document hashes, raw artifacts and a machine-readable audit
are exported locally. Historical Stage 1/2 outputs are unchanged.
