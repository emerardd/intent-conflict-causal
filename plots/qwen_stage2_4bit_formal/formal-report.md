# Formal experiment report

Model: `Qwen/Qwen3.5-4B`
Quantization: `4bit_nf4`
Behavior gate passed: **True**
Plot status: generated as dependency-free SVG files

## Behavioral positive control

- Authorized accuracy: 1.000
- Unauthorized accuracy: 1.000
- Paired margin effect: 9.7914
- Positive scenario fraction: 1.000

## Held-out representation

- Validation-selected layer: 16
- Test AUROC: 1.000
- Train/test direction cosine: 0.997
- Cross-dot permutation p-value: 0.0002

## Causal activation swap

- Authorized→unauthorized raw margin delta: -0.9859
- Unauthorized→authorized raw margin delta: 0.5109
- Mean sign-aligned causal effect: 0.7484
- 95% scenario-cluster bootstrap CI: [0.7047, 0.8000]
- Random-control mean: 0.0473

## Decision

The preregistered pilot supports a cross-scenario, causally active pre-action representation of whether the exact candidate action is authorized. Replication on an unquantized model is warranted.

The study concerns a functional pre-action authorization representation. It does not establish consciousness or natural deceptive intent.