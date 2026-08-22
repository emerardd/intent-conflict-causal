# Formal experiment report

Model: `models/ministral3-3b-instruct-2512`
Quantization: `checkpoint_fp8_dequant_bf16`
Behavior gate passed: **True**
Plot status: generated as dependency-free SVG files

## Behavioral positive control

- Authorized accuracy: 1.000
- Unauthorized accuracy: 1.000
- Paired margin effect: 7.3937
- Positive scenario fraction: 1.000

## Held-out representation

- Validation-selected layer: 15
- Test AUROC: 1.000
- Train/test direction cosine: 0.998
- Cross-dot permutation p-value: 0.0002

## Causal activation swap

- Authorized→unauthorized raw margin delta: -4.2719
- Unauthorized→authorized raw margin delta: 4.0437
- Mean sign-aligned causal effect: 4.1578
- 95% scenario-cluster bootstrap CI: [3.9468, 4.3649]
- Random-control mean: -0.0373

## Decision

The preregistered pilot supports a cross-scenario, causally active pre-action representation of whether the exact candidate action is authorized. Replication on an unquantized model is warranted.

The study concerns a functional pre-action authorization representation. It does not establish consciousness or natural deceptive intent.