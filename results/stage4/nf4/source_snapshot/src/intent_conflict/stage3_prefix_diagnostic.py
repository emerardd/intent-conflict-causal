"""Post-hoc instrumentation check only: no behavioral or mechanistic hypothesis test."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .stage3 import dump, load_model, now, read, sha
from .stage3_model import evaluate


def main():
    root = Path(__file__).resolve().parents[2]
    run = root / "results/stage3/screen-v1"
    destination = run / "prefix_diagnostic.json"
    if destination.exists():
        raise FileExistsError(destination)
    base = read(run / "baseline.json")
    prompts = read(run / "prompts.json")
    p0, p1 = prompts[0], prompts[2]  # Same authorization/mapping; different controller policy.
    k = p0["positions"]["pre_policy"]
    assert p0["ids"][:k+1] == p1["ids"][:k+1]
    loaded = load_model(base["config"])
    original_ids = p0["ids"]
    alternative_tail = p1["ids"][k+1:]
    need = len(original_ids)-k-1
    matched_length = original_ids[:k+1] + (alternative_tail + [loaded.execute_token_id]*need)[:need]
    cases = {"original": original_ids, "other_policy_length": p1["ids"],
             "other_suffix_same_length": matched_length, "prefix_only": original_ids[:k+1]}
    states, logits = {}, {}
    for name, ids in cases.items():
        logits[name], states[name] = evaluate(loaded, ids, {"probe": k}, base["config"]["layers"])
    report = {"created_at_utc": now(), "classification": "post-hoc instrumentation diagnostic; not new formal data",
        "script_sha256": sha(Path(__file__)), "original_baseline_margin_error": abs(logits["original"]["margin"]-base["rows"][0]["margin"]),
        "original_baseline_argmax_agrees": logits["original"]["top1_id"] == base["rows"][0]["top1_id"],
        "lengths": {name: len(ids) for name, ids in cases.items()}, "comparisons": {}}
    reference = states["original"][0]
    for name, state in states.items():
        delta = state[0]-reference
        report["comparisons"][name] = {"max_absolute_difference": float(np.abs(delta).max()),
            "relative_l2_difference": float(np.linalg.norm(delta)/np.linalg.norm(reference)),
            "per_layer_max_abs": {str(layer): float(np.abs(delta[j]).max()) for j, layer in enumerate(base["config"]["layers"])}}
    dump(destination, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
