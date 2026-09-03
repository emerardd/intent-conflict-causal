# Pre-action user-authorization conflict

*[中文版 README](README.zh-CN.md) · [Documentation index](docs/README.md)*

Does a language model internally represent **whether the user authorized the
specific action it is about to take** — and if so, is that representation
*causally load-bearing* for the decision it makes?

This repository holds a five-stage, pre-registered interpretability study of
that question on small open-weight chat models. Its main result is a negative
one, and the reason it exists:

> **Readable is not the same as causally load-bearing.**
> Authorization status is linearly decodable from a single residual-stream
> position with held-out AUROC 1.00 — yet freezing that direction and patching
> it into unauthorized trials moves the decision by a statistically detectable
> but behaviorally negligible amount, and never flips a single decision.

Every claim below is bounded by what the experiments actually license. The
narrow scope is deliberate, and the limits are stated in each stage report.

## The experimental design

One variable is manipulated, and only one:

> With the candidate action text held completely fixed, did the user authorize
> **this specific action**?

Each prompt contains one permission and one prohibition. The two conditions
swap only the binding between *permitted / forbidden* and the *candidate /
distractor* action, so identical action content and identical authorization
vocabulary appear on both sides. The model is measured just before it emits a
single-token decision: `A` (proceed) or `B` (block).

The evidence chain the design targets:

1. **Behavioral positive control** — `log p(B) − log p(A)` is higher in the
   unauthorized condition than in the authorized one.
2. **Representational generalization** — the unauthorized-minus-authorized
   direction fitted on training scenarios keeps its sign on fully held-out
   scenarios.
3. **Causal test** — patching authorized-condition pre-action activations into
   the unauthorized condition should lower the tendency to block, and the
   reverse patch should raise it.
4. **Specificity controls** — candidate-action position is balanced across the
   permission and prohibition clauses, and random equal-norm patches must not
   reproduce the main effect.

The project does **not** train the base model, does **not** treat probe AUC on
its own as mechanistic evidence, and does not study naturally occurring
cheating, action risk, reward hacking, or post-hoc rationalization.

## Results by stage

| Stage | Question | Headline outcome |
| --- | --- | --- |
| **1** | Does the effect exist at all on one model? | Behavioral accuracy 0.993; layer-16 held-out AUROC 1.000; train/test direction cosine 0.977; scenario-clustered permutation `p = 0.001996`; bidirectional patching effect **0.637**, 95% CI `[0.480, 0.766]` vs random-direction mean 0.029. |
| **2** | Does it survive across precision and model family? | Three formal runs, all with test AUROC 1.000 and exact `2^12` scenario sign enumeration `p = 1/4096`: Qwen3.5-4B NF4 effect 0.700 `[0.645, 0.762]`; Qwen3.5-4B BF16 effect 1.268 `[1.205, 1.332]`; Ministral-3-3B BF16 effect 4.158 `[3.947, 4.365]`. 0% decision flips across 160 same-authorization wording swaps per run. |
| **3** | Is the readable direction causally load-bearing? | **No.** With an output-mapping control added, the layer-16 pre-legend direction reaches cross-scenario AUROC 1.00 (0.898 on untrained wording), but 480 frozen interventions flip nothing; main effect **−0.00651**, 95% CI `[−0.01953, 0.00651]` — indistinguishable from random directions. |
| **4** | Is one dimension simply too small a handle? | **Also no.** Even a whole-residual-vector swap at the pre-legend position gives no reliable effect, while the same swap at the *answer* position does move probability preference. Position matters more than dimensionality. Bounded exploratory diagnostic, reusing old scenarios. |
| **5** | Does the position dependence confirm on new scenarios? | 12 new scenarios, 48 baselines, 576 interventions. `answer` local direction minus equal-norm random axis **0.3672** `[0.3021, 0.4297]`; `answer − pre_mapping` interaction 0.3229 `[0.2604, 0.3802]`; `pre_mapping` full-vector −0.0026 `[−0.0339, 0.0234]`, inside the pre-set ±0.1 equivalence bound. All four pre-declared checks passed. |
| **5b** | Is the effect direction-specific, not just norm? | 1,344 interventions with the two real directions and 10 random axes per position frozen to a **strictly equal norm** (0.59587675, taken from Stage 3 training activations). Answer-direction minus random axes **0.1990** `[0.1805, 0.2190]`; minus equal-norm pre-legend direction 0.1771 `[0.1563, 0.1979]`; position × direction-identity interaction 0.1927 `[0.1693, 0.2161]`. All 12/12 scenarios same-sign; independent raw-artifact recomputation passed. |

### What the final result does and does not say

Stage 5b is **controlled, probabilistic causal evidence that the effect is
jointly specific to the answer position and the answer direction.** It is not
evidence of behavioral control, and not evidence that the model "knows it is
violating the user's intent":

- The raw effect recovers only about **2.3%** of the natural margin.
- Across 1,344 interventions there was **not one top-1 flip**.
- The task still binds *unauthorized* to *normatively should block*; the two
  are not separated.
- There is no naturally occurring prompt injection and no real tool behavior.

The full Chinese narrative — evidence evolution, disconfirming results,
retracted claims, limitations and next steps — is in
[docs/project-progress-complete-report.zh-CN.md](docs/project-progress-complete-report.zh-CN.md).

## Repository layout

```
configs/     Experiment configurations, one JSON per run (the unit of provenance)
docs/        Pre-registrations, result reports, reproduction guides, run logs
docs/figures/  Curated figures embedded in the reports
scripts/     Standalone CPU audits that re-derive stored evidence independently
src/intent_conflict/   Library: data, model, experiment, per-stage analysis
tests/       Unit tests (41, CPU-only, no model weights required)
```

`results/` **is tracked**: prompts, baselines, interventions, activations,
frozen axes, environment dumps, source snapshots and manifest hashes all ship
with the repository, so `scripts/audit_stage*.py` runs on a fresh clone. See
[docs/output-layout.md](docs/output-layout.md) for what each artifact contains.
The regenerable figure directory (`plots/`) and model weights (`models/`) are
excluded.

## Setup

Python 3.11+ is required. The interventions need a CUDA GPU; the tests and the
audit scripts do not.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify the install with the CPU-only test suite:

```bash
pytest -q
```

## Running experiments

Stage 1 and Stage 2 go through the CLI, with a config file as the unit of
provenance:

```bash
intent-conflict run        --config configs/smoke.json
intent-conflict run        --config configs/qwen_formal.json
intent-conflict robustness --config configs/qwen_formal.json
intent-conflict reanalyze  --config configs/qwen_formal.json
```

The Stage 2 cross-run synthesis:

```bash
python -m intent_conflict.stage2_synthesis --root .
```

Stages 3–5b are phase-driven modules, run from the repository root:

```bash
python -m intent_conflict.stage3_remap --phase screen
python -m intent_conflict.stage4  --precision nf4
python -m intent_conflict.stage5  --phase run
python -m intent_conflict.stage5b --phase run
```

Each stage has a reproduction guide with the exact commands, the pinned model
revision, and the environment boundary that applies to it:
[Stage 3](docs/stage3-reproduction.md) ·
[Stage 4](docs/stage4-reproduction.md) ·
[Stage 5](docs/stage5-reproduction.md) ·
[Stage 5b](docs/stage5b-reproduction.md).

### Continuation gate

Behavioral screening proceeds to formal patching only if **all** of these hold:

- decision accuracy is at least 80% in both the authorized and unauthorized
  conditions;
- the mean unauthorized-minus-authorized behavioral margin is positive;
- at least 3 of 4 screening scenarios show the predicted direction;
- no degenerate policy that answers `BLOCK` (or `PROCEED`) to both conditions.

Stage 3's v1 factorial screen failed this gate and was stopped at the
threshold rather than being tuned into significance. That failure is kept in
the record — see [docs/stage3-run-log.md](docs/stage3-run-log.md).

## Reproducing the stored evidence

`scripts/audit_stage*.py` re-derive the statistics from stored raw artifacts
**without importing the analysis code**, so an audit failure is a genuine
inconsistency rather than a shared bug. They need only Python 3.11+ and NumPy —
no GPU and no model weights:

```bash
python scripts/audit_stage3.py
python scripts/audit_stage5b.py
```

The raw artifacts these scripts read ship with the repository under
`results/`, so a fresh clone can verify every headline number without a GPU.

> **Note.** Recomputing stored evidence is a weaker claim than rerunning
> inference, and neither one validates the semantic interpretation of a latent
> direction. The audits guard against implementation drift between the analysis
> and the reported numbers, not against a shared conceptual error.

## Models

| Model | Used in | Precision |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | Stages 1–5b (primary) | NF4 and BF16 |
| `Ministral-3-3B-Instruct-2512` | Stage 2 cross-family replication | BF16 |
| `google/gemma-2-2b-it` | Screening only | BF16 |

Stages 4, 5 and 5b pin the Qwen revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in their configs and resolve it
through `snapshot_download(..., local_files_only=True)`, so a run fails rather
than silently falling back to a different revision. The resolved revision is
recorded in each run's manifest.

Weights are not distributed here. Fetch them from the Hub, or point each
config's `model_name` at your own cache — the Ministral config expects a local
directory at `models/ministral3-3b-instruct-2512`.

## License

[MIT](LICENSE).
