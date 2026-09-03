# Stage 4: bounded paired-vector decomposition diagnostic

Locally frozen before Stage 4 outputs, after observing Stage 3. This is an
EXPLORATORY follow-up on already-seen scenarios, not independent confirmation.
No claim of natural violation awareness or authorization/action separation.

Question: does replacing a full single-position residual vector affect the
answer when its train-only authorization-direction component does not? Compare
two fixed positions: before the response legend, and before the answer. Layer
index 16 is fixed from Stage 3, not selected again. The answer position is an
operational comparator, not an authorization-specific positive control.

Keep Qwen3.5-4B and pin the cached checkpoint revision. NF4 uses all twelve
Stage 3 test scenarios, seen grammar and candidate-first order: 48 recipients.
BF16 uses ONLY the first two of those scenes: eight recipients. No larger model,
new prompt, layer search, intensity search, or success-conditioned sample choice.
BF16 is a small precision check, with two scenario clusters, not a full replication.

Fit the direction at each fixed position from all 64 ORIGINAL NF4 training rows:
unauthorized-minus-authorized mean, removing the expected-token contrast if
nonzero, then normalize. Keep these same directions for BF16. Thus BF16 tests
precision of the intervention along a fixed NF4-derived axis, not the best
direction learned independently in BF16. No training of model weights occurs.

Each recipient is paired with the opposite authorization condition, holding
scenario, candidate command, grammar, record order and answer mapping fixed.
Use fresh baseline/donor activations in the current precision. Exact prompt IDs
come from Stage 3; verify dataset identity, equal sequence lengths, and equal
measurement indices within each donor pair. If alignment fails, stop rather than
guessing a token match. Recheck NF4 baseline against its saved Stage 3 result.

At each position define delta = donor - recipient, p = (delta dot d)d, q = delta-p.
Apply exactly one displacement per mode, with no alpha search:

- full: delta (the complete residual vector at ONE position, not the model state);
- parallel: p (paired projection size, not an arbitrary large coefficient);
- perpendicular: q;
- random_full: sign(delta dot d) times ||delta|| times a fixed random unit axis;
- random_parallel: (delta dot d) times that same random axis.

The random axis is orthogonal to d, seeded once per position, and reused for
both donor directions. No semantic specificity from random controls alone.
Perpendicular has no separately norm-matched random control; compare it
descriptively. For exact-zero dot products use positive sign for random_full.

NF4: 48 x 2 positions x 5 modes = 480 intervention forwards, 48 fresh baselines,
plus one null patch per position. BF16: 8 x 2 x 5 = 80 intervention forwards,
eight fresh baselines, plus two null patches. At the first scene, real one-token
greedy generation checks full and parallel patches at both sites (16 per precision).
Observer hooks after each patch record delivered norm/projection, and null patches
must reproduce the fresh baseline. All baseline errors and invalid outputs remain.
If either mapping's baseline accuracy <0.90, save failed baseline and stop that
precision's interventions. Do not replace the failed precision with another model.

Primary descriptive effect = patched-minus-recipient B/A log-probability margin,
signed toward the donor's expected answer. Report mapping-cell and scenario means,
global-top1 flips, flips toward donor, invalid first tokens, and fraction of
natural donor-recipient margin gap recovered where denominator >1e-6. Bootstrap
by the twelve scenarios for NF4 only; do not present two-scene BF16 intervals as
confirmatory evidence. Report paired full-minus-parallel, parallel-minus-random,
full-minus-random and full-minus-perpendicular, not just separate significance.
Nonadditivity is possible: full effect need not equal component effects.

Freeze source/config/reference hashes before baseline inference and preserve all
artifacts. Single-precision estimated upper bound: 15 minutes for loading and
forward work; if resource errors or an overrun occurs, stop and preserve partial
evidence, with no claim of completion. No GPU job is left running after handoff.
No edits to Stage 3 freezes/reports/manifests, no Git commit or push.
