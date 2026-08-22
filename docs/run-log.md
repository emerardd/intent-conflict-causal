# Run log

## 2026-08-15: pipeline smoke

- Model: locally cached `hf-internal-testing/tiny-random-Gemma2ForCausalLM`.
- Result: all code paths, including extraction, split analysis, activation replacement, artifacts, and report generation, completed.
- Scientific status: none; the random model failed the behavioral gate as expected.

## 2026-08-15: Qwen3.5-4B behavior screen v0

- Model: `Qwen/Qwen3.5-4B`, revision resolved from the local cache, 4-bit NF4 with BF16 compute.
- Six scenarios, four counterbalanced variants, two authorization conditions: 48 prompts.
- Unauthorized accuracy: 24/24.
- Authorized accuracy: 19/24 (0.7917), narrowly below the frozen 0.80 gate.
- Every scenario had a positive unauthorized-minus-authorized margin effect; mean paired effect was 7.1198.
- Failure was concentrated: all four authorized variants of `disable_hook` were blocked, while the other five scenarios were 19/20 correct.

Interpretation: the `disable_hook` action carried an independent suspicious-action prior even when explicitly authorized. This contradicted the intended single-variable construct. The threshold was not relaxed. Before any formal representation analysis, the scenario inventory was revised to use risk-neutral simulated administrative actions, while preserving the exact same authorization-binding manipulation. The v0 JSON is retained at `results/qwen_behavior_screen.json`.

## Dataset v1 freeze

- Candidate and distractor actions now have matched, risk-neutral status within each scenario.
- Both conditions still contain one permission and one prohibition.
- Candidate mention position remains fixed within pairs and counterbalanced across variants.
- No formal layer selection or patching result from Qwen was observed before this revision.

## 2026-08-15: Qwen3.5-4B behavior screen v1

- Six risk-neutral scenarios, four counterbalanced variants, two authorization conditions: 48 prompts.
- Authorized accuracy: 23/24 (0.9583).
- Unauthorized accuracy: 24/24 (1.0000).
- Overall accuracy: 47/48 (0.9792).
- Mean paired unauthorized-minus-authorized margin: 9.3906.
- Every screened scenario had a positive paired margin effect.
- Result: all frozen behavioral gates passed.

Before representation analysis, the formal split was frozen explicitly. The six scenarios used during calibration are confined to training. Final test consists only of `push_review_alpha`, `include_record_alpha`, `install_package_alpha`, and `disable_debug_alpha`, none of which had been behaviorally inspected.

## 2026-08-15: Qwen3.5-4B formal run

- Eighteen scenarios, four counterbalanced variants, two conditions: 144 prompts.
- Train/validation/test split by whole scenario: 10/4/4.
- Overall authorized accuracy: 0.9861; unauthorized accuracy: 1.0000.
- Mean paired unauthorized-minus-authorized behavioral margin: 8.9219.
- Final test behavior: 32/32 correct across four previously uninspected scenarios.
- Validation selected layer 16. Final-test AUROC: 1.0000; train/test direction cosine: 0.9768.
- Bidirectional activation replacement at layer 16: authorized-to-unauthorized margin delta -0.7500; unauthorized-to-authorized delta +0.5234; sign-aligned mean 0.6367.
- Five norm-matched random-control means ranged from -0.0156 to 0.0508.

## Statistical correction before finalization

The first permutation implementation normalized every sign-permuted direction. Nearly collinear matched-pair deltas made tiny random residuals saturate direction-only AUROC, yielding an invalid p-value near 0.365. Before final reporting, the statistic was changed to the unnormalized train/test mean-delta dot product. Wording variants were also recognized as repeated measures: sign flips now occur by whole training scenario, and the causal confidence interval bootstraps whole test scenarios. No model forward pass was rerun for this correction; layer selection remained 16.

- Scenario-clustered permutation observed cross-dot: 2.0069.
- Null 95th percentile: 1.0637.
- Monte Carlo p-value with 500 permutations: 0.001996.
- Scenario-clustered causal 95% CI: [0.4805, 0.7656].
- All behavior, representation, and causal support flags remained true.

## Post-hoc exploratory robustness controls

- Wrong-layer control at layer 4: sign-aligned effect 0.0234, about 27 times smaller than the layer-16 main effect; scenario-clustered 95% CI [0.0078, 0.0391].
- Same-condition wording control at layer 16: 32 swaps, mean signed delta -0.0117, mean absolute delta 0.1211, decision-flip rate 0/32.
- These controls were designed after observing the primary result and are explicitly not presented as preregistered evidence.

## 2026-08-15: Stage 2 dataset and preregistration

- Expanded to 40 risk-neutral simulated-action scenarios, with 4 wording/position variants and 2 authorization conditions per scenario: 320 prompts.
- Froze a whole-scenario split of 12 train, 8 validation, and 20 final test scenarios.
- Froze exact enumeration of all `2^12=4096` train-scenario sign assignments and a 20-scenario clustered bootstrap for the causal interval.
- The first Qwen NF4 Stage 2 run used the same `alpha/beta` candidate/distractor words in every scenario. Before Qwen BF16 or a second-family formal result was observed, this was classified as a lexical confound and retained as v1 development evidence.
- Stage 2 v2 assigned a distinct three-character codeword to every candidate and distractor; all 80 codewords are unique across the corpus. Scenario inventory and 12/8/20 split were unchanged.

## 2026-08-15: Qwen3.5-4B Stage 2 v2 NF4

- Final test behavior: 160/160 correct; paired margin effect 10.3469.
- Validation-selected layer 16; test AUROC 1.000; train/test direction cosine 0.9965.
- Exact scenario-sign p-value: `1/4096 = 0.000244140625`.
- Bidirectional full-state swap: authorized-to-unauthorized delta -0.8969; unauthorized-to-authorized delta +0.5031; sign-aligned mean 0.7000; 20-scenario 95% CI `[0.6445, 0.7617]`.
- Five random-direction controls had overall mean 0.0584.
- Post-hoc controls: layer-4 effect 0.0133, CI `[0.0047, 0.0219]`; 160 same-condition swaps had mean absolute delta 0.0711 and 0% decision flips.

## 2026-08-15: Qwen3.5-4B Stage 2 v2 BF16

- Loaded the 8.68 GB checkpoint with decoder layers 0-20 on GPU and 21-31 plus final normalization on CPU; all model parameters reported BF16.
- Final test behavior: 160/160 correct; paired margin effect 11.2875.
- Validation-selected layer 16; test AUROC 1.000; train/test direction cosine 0.9979; exact p `1/4096`.
- Bidirectional full-state sign-aligned mean 1.2680; 95% CI `[1.2055, 1.3320]`; random mean 0.0308.
- Post-hoc controls: layer-4 effect -0.0016, CI `[-0.0141, 0.0109]`; same-condition mean absolute delta 0.1039 and 0% flips.
- NF4/BF16 behavior margins correlate at `r=0.9928`; pair-level causal effects at `r=0.6867`; train authorization directions have cosine 0.8383.

## 2026-08-15: modern second-family selection and Ministral result

- Gemma-2-2B was used only as a pipeline diagnostic and excluded from confirmatory evidence after the user correctly flagged its age.
- Selected `mistralai/Ministral-3-3B-Instruct-2512` as a recent, different-family checkpoint. The official FP8 path downloaded `kernels-community/finegrained-fp8` v4 but could not execute on native Windows because it requires Triton.
- Used the model-card-supported `FineGrainedFP8Config(dequantize=True)` path instead. The loaded parameters were all BF16; decoder layers 0-20 ran on GPU and 21-25 plus final normalization on CPU.
- Six-scenario behavior screen: 48/48 correct, paired margin effect 7.6563.
- Formal final test: 160/160 correct, paired margin effect 7.4781.
- Validation-selected layer 15; test AUROC 1.000; train/test direction cosine 0.9976; exact p `1/4096`.
- Bidirectional full-state sign-aligned mean 4.1578; 95% CI `[3.9468, 4.3649]`; random mean -0.0373.
- Post-hoc controls: layer-3 effect 0.0016, CI `[-0.0063, 0.0102]`; same-condition mean absolute delta 0.2164 and 0% flips.

## 2026-08-15: Stage 2 synthesis audit

- `intent_conflict.stage2_synthesis` checked the frozen dataset version, disjoint 12/8/20 split, 320 unique examples, eight examples per scenario, exact JSON/NPZ example ordering, activation shapes, behavior gates, validation-based layer score, exact enumeration metadata, 80 causal pairs across 20 scenarios, and both robustness controls.
- All audit checks passed for all three formal runs.
- Generated `results/stage2_comparison.json`, three PNG/SVG comparison figures, and `docs/stage2-report.zh-CN.md`.
- Claim boundary: the study supports a cross-scenario explicit-authorization state whose full residual representation is causally relevant. It does not yet show natural self-knowledge of violating latent user intent, deception, consciousness, or causal sufficiency of the one-dimensional probe direction.
