"""Bounded Stage 4 paired residual-vector diagnostic; original Stage 3 untouched."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import importlib.metadata
import platform
from pathlib import Path
import shutil
import time

import numpy as np

from .stage3 import dump, manifest, now, read, sha
from .stage3_analysis import contrast, residualize, unit


def decompose(recipient, donor, direction, random_axis):
    delta = donor.astype(float) - recipient.astype(float)
    coefficient = float(delta @ direction)
    parallel = coefficient * direction
    return {"full": delta, "parallel": parallel, "perpendicular": delta-parallel,
        "random_full": (1 if coefficient >= 0 else -1)*np.linalg.norm(delta)*random_axis,
        "random_parallel": coefficient*random_axis}


def pair_indices(rows):
    lookup = {}
    fields = ("scenario_id", "grammar", "order", "reversed_mapping")
    for i, row in enumerate(rows):
        key = tuple(row[f] for f in fields) + (row["authorized"],)
        if key in lookup:
            raise ValueError("Duplicate donor key")
        lookup[key] = i
    pairs = []
    for i, row in enumerate(rows):
        j = lookup[tuple(row[f] for f in fields)+(not row["authorized"],)]
        if row["candidate_command"] != rows[j]["candidate_command"]:
            raise ValueError("Candidate mismatch")
        pairs.append(j)
    return pairs


def summarize(rows, baselines, bootstrap_samples, seed):
    baseline = {r["example_id"]: r for r in baselines}
    groups = defaultdict(list)
    for row in rows:
        b, donor = baseline[row["example_id"]], baseline[row["donor_id"]]
        sign = 1 if donor["expected_label"] == "B" else -1
        effect = sign*(row["margin"]-b["margin"])
        gap = sign*(donor["margin"]-b["margin"])
        groups[row["position"], row["mode"]].append((row, effect, gap))
    report, scenario_groups = {}, {}
    for (position, mode), group in groups.items():
        scenario = defaultdict(list)
        for row, effect, _ in group:
            scenario[baseline[row["example_id"]]["scenario_id"]].append(effect)
        means = {sid: float(np.mean(v)) for sid, v in scenario.items()}
        scenario_groups[position, mode] = means
        def interval(values):
            if len(values) < 4:
                return None
            rng = np.random.default_rng(seed)
            return np.quantile(rng.choice(values, (bootstrap_samples, len(values)), replace=True).mean(1), [.025, .975]).tolist()
        key = f"{position}/{mode}"
        report[key] = {"mean": float(np.mean(list(means.values()))), "ci95": interval(list(means.values())),
            "n_scenarios": len(means), "n_forwards": len(group), "scenario_effects": means,
            "mapping_means": {str(m): float(np.mean([e for r,e,g in group if baseline[r['example_id']]['reversed_mapping'] == m])) for m in (False, True)},
            "global_top1_flips": sum(r["top1_id"] != baseline[r["example_id"]]["top1_id"] for r,e,g in group),
            "flips_toward_donor": sum(r["top1_id"] != baseline[r["example_id"]]["top1_id"] and r["top1_label"] == baseline[r["donor_id"]]["expected_label"] for r,e,g in group),
            "invalid_first_tokens": sum(r["top1_label"] == "OTHER" for r,e,g in group),
            "mean_gap_recovery": float(np.mean([e/g for r,e,g in group if g > 1e-6])) if any(g > 1e-6 for r,e,g in group) else None,
            "excluded_nonpositive_gap": sum(g <= 1e-6 for r,e,g in group)}
    comparisons = {}
    for position in dict.fromkeys(p for p,m in scenario_groups):
        for left, right in (("full", "parallel"), ("parallel", "random_parallel"), ("full", "random_full"), ("full", "perpendicular")):
            diffs = [v-scenario_groups[position,right][sid] for sid,v in scenario_groups[position,left].items()]
            comparisons[f"{position}/{left}-minus-{right}"] = {"mean": float(np.mean(diffs)), "ci95": interval(diffs)}
    return {"effects": report, "paired_comparisons": comparisons,
        "estimand": "Donor-answer-aligned change in B/A log-probability margin; scenario-level averaging.",
        "scope": "Exploratory paired diagnostic, not semantic specificity or independent confirmation."}


def patched_evaluate(loaded, prompt, position, layer, vector, original, direction):
    from .stage3_model import checked_patch, evaluate
    observed = {}
    with checked_patch(loaded, layer, prompt["positions"][position], vector, len(prompt["ids"])):
        def observe(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            observed["h"] = hidden[0, prompt["positions"][position]].detach().float().cpu().numpy()
        handle = loaded.layers[layer].register_forward_hook(observe)
        try:
            measured, _ = evaluate(loaded, prompt["ids"], prompt["positions"], [])
        finally:
            handle.remove()
    actual = observed["h"]-original
    return {**measured, "requested_delta_norm": float(np.linalg.norm(vector-original)),
        "delivered_delta_norm": float(np.linalg.norm(actual)), "delivered_projection": float(actual @ direction)}


def run(precision):
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/stage4_diagnostic.json"
    config = read(config_path)
    output = root / config["output_root"] / precision
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference = root / config["reference"]
    old = read(reference / "baseline.json")
    prompts = read(reference / "prompts.json")
    archive = np.load(reference / "activations.npz", allow_pickle=False)
    if archive["example_ids"].tolist() != [r["example_id"] for r in old["rows"]] or [p["example_id"] for p in prompts] != archive["example_ids"].tolist():
        raise ValueError("Reference identity mismatch")
    sites = config["positions"]
    layer_idx = archive["layers"].tolist().index(config["layer"])
    train = np.array([r["split"] == "train" for r in old["rows"]])
    auth = np.array([not r["authorized"] for r in old["rows"]])
    token = np.array([r["expected_label"] == "B" for r in old["rows"]])
    axes = {}
    rng = np.random.default_rng(config["seed"])
    for position in sites:
        h = archive["activations"][:, archive["positions"].tolist().index(position), layer_idx]
        raw = contrast(h, auth, train)
        clean, _ = residualize(raw, [contrast(h, token, train)])
        d = unit(clean)
        r = rng.normal(size=d.shape)
        r = unit(r-(r @ d)*d)
        axes[position] = d
        axes[position+"_random"] = r
    sids = list(dict.fromkeys(r["scenario_id"] for r in old["rows"] if r["split"] == "test"))[:config[f"{precision}_test_scenarios"]]
    indices = [i for i,r in enumerate(old["rows"]) if r["scenario_id"] in sids and r["grammar"] == "seen" and r["order"] == "candidate_first"]
    originals = [old["rows"][i] for i in indices]
    prompts = [prompts[i] for i in indices]
    pairs = pair_indices(originals)
    for i,j in enumerate(pairs):
        if len(prompts[i]["ids"]) != len(prompts[j]["ids"]) or any(prompts[i]["positions"][p] != prompts[j]["positions"][p] for p in sites):
            raise ValueError("Donor token alignment failed")
    source = [*sorted((root / "src/intent_conflict").glob("*.py")), config_path, root / "docs/stage4-preregistration.md", root / "tests/test_stage4.py"]
    output.mkdir(parents=True)
    for path in source:
        dest = output / "source_snapshot" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    dump(output / "freeze.json", {"created_at_utc": now(), "config": config, "precision": precision,
        "python": platform.python_version(),
        "source_sha256": {p.relative_to(root).as_posix(): sha(p) for p in source},
        "reference_sha256": {p.name: sha(p) for p in (reference / "baseline.json", reference / "activations.npz", reference / "prompts.json")},
        "recipient_ids": [r["example_id"] for r in originals], "donor_indices": pairs})
    (output / "environment.txt").write_text("\n".join(sorted(f"{p.metadata['Name']}=={p.version}" for p in importlib.metadata.distributions()))+"\n", encoding="utf-8")
    dump(output / "prompts.json", prompts)
    np.savez_compressed(output / "axes.npz", **axes)
    import torch
    from huggingface_hub import snapshot_download
    from .model import LoadedModel
    from .stage3_model import evaluate, generate
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = snapshot_download(config["model_name"], revision=config["revision"], local_files_only=True)
    started = time.monotonic()
    print(f"Loading cached {precision} model from pinned snapshot...", flush=True)
    loaded = LoadedModel.load(model_name=snapshot, model_loader="multimodal_lm",
        device="cuda" if precision == "nf4" else "auto", dtype="bfloat16", local_files_only=True,
        allow_fallback_chat_template=False, quantization="4bit_nf4" if precision == "nf4" else "none",
        max_memory=None if precision == "nf4" else {"0": "6.5GiB", "cpu": "6GiB"},
        offload_folder=None if precision == "nf4" else str(output / "offload"))
    dump(output / "provenance.json", {"created_at_utc": now(), "snapshot": snapshot,
        "revision": config["revision"], "model_class": type(loaded.model).__name__,
        "A_id": loaded.execute_token_id, "B_id": loaded.block_token_id,
        "parameter_dtypes": sorted({str(p.dtype) for p in loaded.model.parameters()}),
        "device_map": {str(k): str(v) for k,v in getattr(loaded.model, "hf_device_map", {}).items()}})
    def deadline():
        if time.monotonic()-started > 900:
            raise TimeoutError("Frozen 15-minute per-precision bound exceeded; partial evidence preserved")
    baselines, hidden = [], []
    with (output / "baseline-progress.jsonl").open("x", encoding="utf-8") as journal:
        for i,prompt in enumerate(prompts):
            deadline()
            positions = {p: prompt["positions"][p] for p in sites}
            measured, h = evaluate(loaded, prompt["ids"], positions, [config["layer"]])
            row = {**originals[i], **measured}
            row["observed_block"] = ((row["top1_label"] == "B") != row["reversed_mapping"]) if row["top1_label"] in {"A", "B"} else None
            row.pop("generated_ids", None)
            row.pop("generated_text", None)
            if precision == "nf4" and (abs(row["margin"]-originals[i]["margin"]) > 1e-4 or row["top1_id"] != originals[i]["top1_id"]):
                raise ValueError("NF4 baseline failed to reproduce Stage 3")
            baselines.append(row)
            hidden.append(h[:, 0])
            journal.write(json.dumps(row)+"\n")
            journal.flush()
            if (i+1)%8 == 0:
                print(f"baseline {i+1}/{len(prompts)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    x = np.stack(hidden)
    dump(output / "baseline.json", {"rows": baselines})
    np.savez_compressed(output / "activations.npz", activations=x, example_ids=np.array([r["example_id"] for r in baselines]), positions=np.array(sites))
    accuracies = {str(m): np.mean([r["top1_label"] == r["expected_label"] for r in baselines if r["reversed_mapping"] == m]) for m in (False, True)}
    if min(accuracies.values()) < config["baseline_min_accuracy"]:
        dump(output / "stopped.json", {"reason": "baseline accuracy gate failed", "mapping_accuracy": accuracies})
        return
    interventions, generated, nulls = [], [], []
    with (output / "interventions-progress.jsonl").open("x", encoding="utf-8") as journal:
        for i,(b,prompt) in enumerate(zip(baselines,prompts)):
            for p,position in enumerate(sites):
                d, r = axes[position], axes[position+"_random"]
                original, donor = x[i,p], x[pairs[i],p]
                if i == 0:
                    null = patched_evaluate(loaded, prompt, position, config["layer"], original, original, d)
                    if abs(null["margin"]-b["margin"]) > 1e-4 or null["top1_id"] != b["top1_id"]:
                        raise ValueError("Null patch differs from fresh baseline")
                    nulls.append({"position": position, **null})
                components = decompose(original, donor, d, r)
                for mode in config["modes"]:
                    deadline()
                    vector = original.astype(float)+components[mode]
                    measured = patched_evaluate(loaded, prompt, position, config["layer"], vector, original, d)
                    item = {"example_id": b["example_id"], "donor_id": baselines[pairs[i]]["example_id"], "position": position, "mode": mode, **measured}
                    if b["scenario_id"] == sids[0] and mode in {"full", "parallel"}:
                        ids, text = generate(loaded, prompt["ids"], 1, patch=(config["layer"], prompt["positions"][position], vector))
                        generated.append({"example_id": b["example_id"], "position": position, "mode": mode,
                            "generated_ids": ids, "generated_text": text, "matches_forward": ids[0] == measured["top1_id"]})
                    interventions.append(item)
                    journal.write(json.dumps(item)+"\n")
                    journal.flush()
            if (i+1)%4 == 0:
                print(f"patch {i+1}/{len(baselines)} recipients, {len(interventions)} forwards, elapsed={time.monotonic()-started:.1f}s", flush=True)
    result = {"completed_at_utc": now(), "elapsed_seconds": time.monotonic()-started,
        "mapping_accuracy": accuracies, "rows": interventions, "null_checks": nulls,
        "generation_audit": generated, "generation_agreement": all(r["matches_forward"] for r in generated),
        "summary": summarize(interventions, baselines, config["bootstrap_samples"], config["seed"]+1)}
    dump(output / "interventions.json", result)
    manifest(output)
    print(json.dumps({"precision": precision, "generation_agreement": result["generation_agreement"], "summary": result["summary"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("nf4", "bf16"), required=True)
    args = parser.parse_args()
    run(args.precision)


if __name__ == "__main__":
    main()
