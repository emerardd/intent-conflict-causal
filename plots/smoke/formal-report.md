# Formal experiment report

Model: `hf-internal-testing/tiny-random-Gemma2ForCausalLM`
Quantization: `none`
Behavior gate passed: **False**
Plot status: matplotlib unavailable; numerical artifacts and Markdown report generated

## Behavioral positive control

- Authorized accuracy: 1.000
- Unauthorized accuracy: 0.000
- Paired margin effect: -0.0000
- Positive scenario fraction: 0.333

## Held-out representation

- Validation-selected layer: 0
- Test AUROC: 0.500
- Train/test direction cosine: 0.086
- Permutation p-value: 1.0000

## Causal activation swap

- Authorized→unauthorized raw margin delta: 0.0000
- Unauthorized→authorized raw margin delta: -0.0000
- Mean sign-aligned causal effect: -0.0000
- 95% paired bootstrap CI: [-0.0013, 0.0013]
- Random-control mean: -0.0060

## Decision

The smoke pipeline ran, but the behavioral positive-control gate failed; the mechanistic numbers are execution checks only.

The study concerns a functional pre-action authorization representation. It does not establish consciousness or natural deceptive intent.