"""Post-hoc instrumentation check only: did the frozen intervention get applied?

Exactly four forwards on the FIRST frozen recipient: baseline, null, -1 and +1.
No new site, strength, recipient selection, or scientific hypothesis test.
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_conflict.stage3 import dump, load_model, now, read, sha
from intent_conflict.stage3_model import checked_patch, evaluate


def main():
    run = ROOT / "results/stage3/remapping-v2/formal-v1"
    target = run / "delivery_diagnostic.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite diagnostic")
    base, plan = read(run / "baseline.json"), read(run / "intervention_freeze.json")
    eid = plan["recipient_ids"][0]
    i = [r["example_id"] for r in base["rows"]].index(eid)
    row = base["rows"][i]
    prompt = read(run / "prompts.json")[i]
    archive = np.load(run / "selected_site.npz", allow_pickle=False)
    h, direction = archive["activations"][i], archive["auth_clean"]
    layer, position, scale = plan["layer"], prompt["positions"][plan["position"]], plan["scale"]
    loaded = load_model(base["config"])
    recorded = read(run / "interventions.json")["rows"]
    cases = {}
    for name, sign in (("baseline", None), ("null", 0), ("minus_one", -1), ("plus_one", 1)):
        observed = {}
        vector = h.astype(float) + (sign or 0)*scale*direction
        context = nullcontext() if sign is None else checked_patch(loaded, layer, position, vector, len(prompt["ids"]))

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            observed["hidden"] = hidden[0, position].detach().float().cpu().numpy()
            observed["hidden_dtype"] = str(hidden.dtype)

        def capture_logits(_module, _inputs, output):
            observed["logits_dtype"] = str(output.logits.dtype)

        with context:
            # Register AFTER the patch hook, so this sees the delivered vector.
            handle = loaded.layers[layer].register_forward_hook(capture)
            out_handle = loaded.model.register_forward_hook(capture_logits)
            try:
                measured, _ = evaluate(loaded, prompt["ids"], prompt["positions"], [])
            finally:
                handle.remove()
                out_handle.remove()
        delta = observed.pop("hidden") - h
        expected = row if sign in (None, 0) else next(r for r in recorded if r["example_id"] == eid and r["axis"] == "auth_clean" and r["alpha"] == 1 and r["sign"] == sign)
        error = abs(measured["margin"] - expected["margin"])
        assert error < 1e-4 and measured["top1_id"] == expected["top1_id"]
        if sign not in (None, 0):
            assert np.linalg.norm(delta) > 0
            assert np.dot(delta, direction)*sign > 0
        cases[name] = {**observed, "actual_delta_norm": float(np.linalg.norm(delta)),
            "requested_delta_norm": float(abs(sign or 0)*scale),
            "actual_direction_projection": float(np.dot(delta, direction)),
            "baseline_hidden_norm": float(np.linalg.norm(h)),
            "margin": measured["margin"], "margin_error_vs_saved": error,
            "top1_label": measured["top1_label"], "top1_matches_saved": True}
    result = {"created_at_utc": now(), "script_sha256": sha(Path(__file__)),
        "classification": "post-hoc implementation delivery diagnostic, not a new hypothesis test",
        "recipient_id": eid, "layer": layer, "position": position, "forwards": 4, "cases": cases,
        "scope": "Checks that nonzero perturbations survive dtype casting and reproduce saved outputs. Does not establish a behavior-changing positive control or exclude redundancy."}
    dump(target, result)
    print(target)
    print(cases)


if __name__ == "__main__":
    main()
