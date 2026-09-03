"""Stage 3 CLI. Run screen -> formal -> intervene; verify is read-only.

python -m intent_conflict.stage3 --phase screen --config configs/stage3_factorial.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .stage3_data import build_trials, tokenize_trial, render_trial, observed_block
from .stage3_analysis import (auc, behavior_summary, bootstrap_ci, evaluate_site, mask_for,
    projection_summary, select_site, summarize_interventions, target, unit)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_files(root: Path) -> list[Path]:
    files = list((root / "src/intent_conflict").glob("*.py"))
    files += [root / "configs/stage3_factorial.json", root / "docs/stage3-preregistration.md"]
    files += list((root / "tests").glob("test_stage3*.py"))
    return sorted(files)


def load_model(config: dict):
    import torch
    from .model import LoadedModel
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    print("Loading cached model (offline)...", flush=True)
    loaded = LoadedModel.load(model_name=config["model_name"], model_loader=config["model_loader"],
        device=config["device"], dtype=config["dtype"], local_files_only=True,
        allow_fallback_chat_template=False, quantization=config["quantization"])
    return loaded


def audit_prefixes(trials, prompt_ids, positions) -> dict:
    references = {}
    counts = defaultdict(int)
    for trial, ids, pos in zip(trials, prompt_ids, positions, strict=True):
        for name in ("pre_policy", "pre_mapping"):
            key = (name, trial.scenario_id, trial.grammar, trial.order, trial.authorized)
            if name == "pre_mapping":
                key += (trial.reversed_policy,)
            prefix = ids[:pos[name]+1]
            if key in references and references[key] != prefix:
                raise ValueError(f"Future policy/mapping leaked into prefix: {trial.example_id} {name}")
            references[key] = prefix
            counts[name] += 1
    return {"passed": True, "groups": len(references), "prefixes_checked": dict(counts)}


def numerical_prefix_audit(rows: list[dict], x: np.ndarray, positions: list[str]) -> dict:
    reports = {}
    for p in ("pre_policy", "pre_mapping"):
        idx = positions.index(p)
        groups = {}
        maxabs, maxrelative = 0.0, 0.0
        for row, hidden in zip(rows, x[:, idx], strict=True):
            key = (row["scenario_id"], row["grammar"], row["order"], row["authorized"])
            if p == "pre_mapping":
                key += (row["reversed_policy"],)
            if key in groups:
                delta = hidden - groups[key]
                maxabs = max(maxabs, float(np.abs(delta).max()))
                maxrelative = max(maxrelative, float(np.linalg.norm(delta) / max(np.linalg.norm(groups[key]), 1e-12)))
            else:
                groups[key] = hidden
        reports[p] = {"max_absolute_difference": maxabs, "max_relative_l2_difference": maxrelative,
                      "interpretation": "same causal prefix; floating-point kernel-length variation may be nonzero"}
    return reports


def inference(config: dict, config_path: Path, phase: str, root: Path) -> Path:
    output = root / config["output_root"] / f"{phase}-v1"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment directory: {output}")
    if phase == "formal":
        screen = read(root / config["output_root"] / "screen-v1/screen_summary.json")
        if not screen["passed"] or screen["config_sha256"] != sha(config_path):
            raise RuntimeError("Pilot failed or config changed after pilot; formal run is forbidden")
    trials = build_trials(phase, config["seed"])
    output.mkdir(parents=True)
    hashes = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in source_files(root)}
    for p in source_files(root):
        dest = output / "source_snapshot" / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    freeze = {"frozen_at_utc": now(), "phase": phase, "config": config,
        "config_sha256": sha(config_path), "source_sha256": hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "python": platform.python_version(), "dataset_rows": [t.row() for t in trials]}
    dump(output / "freeze.json", freeze)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (output / "environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    loaded = load_model(config)
    if max(config["layers"]) >= len(loaded.layers)-1:
        raise ValueError("Exclude final decoder block to avoid final-norm ambiguity")
    prompt_ids, positions = [], []
    for t in trials:
        ids, pos = tokenize_trial(loaded.tokenizer, t)
        prompt_ids.append(ids)
        positions.append({p: pos[p] for p in config["positions"]})
    prefix_audit = audit_prefixes(trials, prompt_ids, positions)
    token_records = [{"example_id": t.example_id, "ids": ids, "positions": pos,
        "position_tokens": {p: loaded.tokenizer.decode(ids[max(0, n-3):n+1]) for p, n in pos.items()},
        "messages": render_trial(t)} for t, ids, pos in zip(trials, prompt_ids, positions, strict=True)]
    dump(output / "prompts.json", token_records)
    provenance = {"inference_started_at_utc": now(), "model_name": config["model_name"],
        "model_revision": getattr(loaded.model.config, "_commit_hash", None),
        "model_class": type(loaded.model).__name__, "quantization": config["quantization"],
        "parameter_dtypes": sorted({str(p.dtype) for p in loaded.model.parameters()}),
        "device_map": {str(k): str(v) for k, v in getattr(loaded.model, "hf_device_map", {}).items()},
        "A_id": loaded.execute_token_id, "B_id": loaded.block_token_id,
        "prefix_audit": prefix_audit, "prompts_sha256": sha(output / "prompts.json")}
    dump(output / "provenance.json", provenance)
    from .stage3_model import evaluate, generate
    first_scenario = {}
    for t in trials:
        first_scenario.setdefault(t.split, t.scenario_id)
    rows, activations = [], []
    start = time.monotonic()
    with (output / "progress.jsonl").open("x", encoding="utf-8") as journal:
        for index, (t, ids, pos) in enumerate(zip(trials, prompt_ids, positions, strict=True)):
            measured, hidden = evaluate(loaded, ids, pos, config["layers"])
            row = {**t.row(), **measured, "prompt_token_count": len(ids), "positions": pos,
                   "observed_block": observed_block(measured["top1_label"], t.reversed_mapping)}
            if phase == "screen" or t.scenario_id == first_scenario[t.split]:
                generated_ids, generated_text = generate(loaded, ids, config["generation_max_new_tokens"])
                row.update(generated_ids=generated_ids, generated_text=generated_text)
            rows.append(row)
            activations.append(hidden)
            journal.write(json.dumps(row, ensure_ascii=False) + "\n")
            journal.flush()
            if (index+1) % 16 == 0:
                elapsed = time.monotonic()-start
                print(f"{phase}: {index+1}/{len(trials)}, elapsed={elapsed:.1f}s, ETA={elapsed/(index+1)*(len(trials)-index-1):.1f}s", flush=True)
    x = np.stack(activations)
    np.savez_compressed(output / "activations.npz", activations=x,
        example_ids=np.asarray([t.example_id for t in trials]), layers=config["layers"], positions=config["positions"])
    behavior = behavior_summary(rows)
    result = {"completed_at_utc": now(), "elapsed_seconds": time.monotonic()-start,
        "config": config, "config_sha256": sha(config_path), "rows": rows, "behavior": behavior,
        "causal_prefix_numerical_audit": numerical_prefix_audit(rows, x, config["positions"])}
    dump(output / "baseline.json", result)
    if phase == "screen":
        passed = (behavior["screen/seen"]["min_cell_top1_accuracy"] >= config["screen_min_cell_top1_accuracy"] and
                  behavior["generation"]["exact_accuracy"] >= config["generation_min_exact_accuracy"] and
                  behavior["generation"]["first_token_agreement"])
        summary = {"passed": passed, "config_sha256": sha(config_path), "behavior": behavior,
                   "stop_reason": None if passed else "Frozen pilot gate failed; no formal inference allowed"}
        dump(output / "screen_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {output}", flush=True)
    return output


def reference_transfer(root: Path, config: dict, rows: list[dict], archive) -> dict:
    reference = root / config["stage2_reference"]
    raw = read(reference)
    old_archive = np.load(reference.with_name(reference.stem + "_activations.npz"), allow_pickle=False)
    old_rows = raw["behavior_rows"]
    if old_archive["example_ids"].tolist() != [r["example_id"] for r in old_rows]:
        raise ValueError("Stage 2 reference order mismatch")
    x = old_archive["activations"][:, old_archive["layers"].tolist().index(16)].astype(np.float64)
    train = np.asarray([r["scenario_id"] in raw["split"]["train"] for r in old_rows])
    authorized = np.asarray([r["authorized"] for r in old_rows])
    direction = unit(x[train & ~authorized].mean(0)-x[train & authorized].mean(0))
    from .stage3_analysis import cell_aucs
    offset = config["layers"].index(16)
    return {position: {grammar: cell_aucs(rows, archive[:, i, offset] @ direction, target(rows, "auth"), "test", grammar)
                       for grammar in ("seen", "unseen")}
            for i, position in enumerate(config["positions"])}


def analyze(output: Path, root: Path, write: bool = True) -> dict:
    result = read(output / "baseline.json")
    config, rows = result["config"], result["rows"]
    archive = np.load(output / "activations.npz", allow_pickle=False)
    if archive["example_ids"].tolist() != [r["example_id"] for r in rows]:
        raise ValueError("Archive example IDs do not match rows")
    if archive["positions"].tolist() != config["positions"] or archive["layers"].tolist() != config["layers"]:
        raise ValueError("Archive position/layer identity mismatch")
    x = archive["activations"]
    site_reports, directions_by_site = [], {}
    for p, position in enumerate(config["positions"]):
        for l, layer in enumerate(config["layers"]):
            report, directions = evaluate_site(rows, x[:, p, l])
            report.update(position=position, layer=layer)
            site_reports.append(report)
            directions_by_site[position, layer] = directions
    selected = select_site(site_reports, config["primary_position"])
    directions = directions_by_site[selected["position"], selected["layer"]]
    selected_x = x[:, config["positions"].index(selected["position"]), config["layers"].index(selected["layer"])]
    scores = selected_x @ directions["auth_clean"]
    behavior = behavior_summary(rows)
    training_behavior_ok = all(behavior[f"{s}/seen"]["min_cell_top1_accuracy"] >= config["formal_min_cell_top1_accuracy"] for s in ("train", "validation"))
    test_behavior_ok = behavior["test/seen"]["min_cell_top1_accuracy"] >= config["formal_min_cell_top1_accuracy"]
    generation_ok = behavior["generation"]["exact_accuracy"] >= config["generation_min_exact_accuracy"] and behavior["generation"]["first_token_agreement"]
    gates = {"train_validation_behavior": training_behavior_ok, "test_seen_behavior": test_behavior_ok,
        "generation": generation_ok, "validation_representation": selected["selection_score"] > config["validation_min_auroc"],
        "test_representation": min([*selected["test_seen_cells"].values(), *selected["test_seen_by_assigned_action"].values()]) > config["test_min_auroc"]}
    summary = {"config": config, "behavior": behavior, "sites": site_reports, "selected": selected,
        "projection": projection_summary(rows, scores, config["seed"]+10, config["bootstrap_samples"]),
        "gates": gates, "intervention_eligible": all(gates.values()),
        "stage2_direction_transfer": reference_transfer(root, config, rows, x),
        "causal_prefix_numerical_audit": result["causal_prefix_numerical_audit"],
        "claim_boundary": "Controlled simulation: independent of assigned controller action and output mapping, not natural violation awareness."}
    if write:
        dump(output / "analysis.json", summary)
        np.savez_compressed(output / "selected_site.npz", activations=selected_x,
            example_ids=archive["example_ids"], auth_scores=scores, layer=selected["layer"], position=selected["position"],
            **directions)
    return summary


def intervene(output: Path, root: Path) -> dict:
    summary = read(output / "analysis.json")
    if not summary["intervention_eligible"]:
        return {"status": "skipped", "reason": "frozen eligibility gate failed", "gates": summary["gates"]}
    if (output / "interventions.json").exists() or (output / "interventions-progress.jsonl").exists():
        raise FileExistsError("Refusing to overwrite interventions")
    config = summary["config"]
    baseline = read(output / "baseline.json")["rows"]
    prompts = read(output / "prompts.json")
    archive = np.load(output / "selected_site.npz", allow_pickle=False)
    if archive["example_ids"].tolist() != [r["example_id"] for r in baseline] or [p["example_id"] for p in prompts] != archive["example_ids"].tolist():
        raise ValueError("Intervention identity mismatch")
    current_hashes = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in source_files(root)}
    if current_hashes != read(output / "freeze.json")["source_sha256"]:
        raise RuntimeError("Source changed since inference freeze; refuse unregistered intervention")
    layer = summary["selected"]["layer"]
    position = config["primary_position"]
    scale = summary["selected"]["train_projection_gap"]
    if scale <= 0:
        raise ValueError("Nonpositive training intervention scale")
    axes = {"auth_clean": archive["auth_clean"], "action": archive["action"], "token": archive["token"]}
    axes = {k: v for k, v in axes.items() if np.linalg.norm(v) > 0.5}
    rng = np.random.default_rng(config["seed"]+20)
    for j in range(config["random_controls"]):
        v = rng.normal(size=axes["auth_clean"].shape)
        v -= (v @ axes["auth_clean"])*axes["auth_clean"]
        axes[f"random{j}"] = unit(v)
    chosen = [i for i, r in enumerate(baseline) if r["split"] == "test" and r["grammar"] == "seen" and r["order"] == "candidate_first"]
    frozen_plan = {"created_at_utc": now(), "layer": layer, "position": position, "scale": scale,
        "recipient_ids": [baseline[i]["example_id"] for i in chosen], "axes": list(axes), "config": config,
        "source_sha256": current_hashes}
    dump(output / "intervention_freeze.json", frozen_plan)
    np.savez_compressed(output / "intervention_axes.npz", **axes)
    loaded = load_model(config)
    from .stage3_model import evaluate, generate
    result_rows, generated_audit = [], []
    started = time.monotonic()
    first_sid = baseline[chosen[0]]["scenario_id"]
    with (output / "interventions-progress.jsonl").open("x", encoding="utf-8") as journal:
        for n, idx in enumerate(chosen):
            row, prompt = baseline[idx], prompts[idx]
            ids, positions = prompt["ids"], prompt["positions"]
            h = archive["activations"][idx]
            if n == 0:
                null, _ = evaluate(loaded, ids, positions, [], patch=(layer, positions[position], h))
                if abs(null["margin"]-row["margin"]) > 1e-4 or null["top1_id"] != row["top1_id"]:
                    raise RuntimeError("Null patch did not reproduce baseline")
            for axis, direction in axes.items():
                alphas = config["alphas"] if axis == "auth_clean" else [1.0]
                for alpha in alphas:
                    for sign in (-1, 1):
                        vector = h.astype(np.float64) + sign*alpha*scale*direction
                        patch = (layer, positions[position], vector)
                        measured, _ = evaluate(loaded, ids, positions, [], patch=patch)
                        item = {"example_id": row["example_id"], "scenario_id": row["scenario_id"],
                            "axis": axis, "alpha": alpha, "sign": sign, **measured}
                        if row["scenario_id"] == first_sid and axis == "auth_clean" and alpha == 1.0:
                            gids, gtext = generate(loaded, ids, 1, patch=patch)
                            generated_audit.append({"example_id": row["example_id"], "sign": sign,
                                "generated_ids": gids, "generated_text": gtext, "matches_forward": gids[0] == measured["top1_id"]})
                        result_rows.append(item)
                        journal.write(json.dumps(item, ensure_ascii=False)+"\n")
                        journal.flush()
            if (n+1) % 8 == 0:
                elapsed = time.monotonic()-started
                print(f"intervention: {n+1}/{len(chosen)} recipients, {len(result_rows)} forwards, elapsed={elapsed:.1f}s, ETA={elapsed/(n+1)*(len(chosen)-n-1):.1f}s", flush=True)
    stats = summarize_interventions(result_rows, {r["example_id"]: r for r in baseline}, config["seed"]+30, config["bootstrap_samples"])
    result = {"completed_at_utc": now(), "elapsed_seconds": time.monotonic()-started, "rows": result_rows,
              "summary": stats, "generation_audit": generated_audit, "generation_agreement": all(r["matches_forward"] for r in generated_audit)}
    dump(output / "interventions.json", result)
    return result


def verify(output: Path, root: Path) -> dict:
    saved = read(output / "analysis.json")
    fresh = analyze(output, root, write=False)
    if json.dumps(saved, sort_keys=True) != json.dumps(fresh, sort_keys=True):
        raise AssertionError("Recomputed analysis does not match")
    base = read(output / "baseline.json")
    expected_rows = [t.row() for t in build_trials("formal", base["config"]["seed"])]
    for expected, observed in zip(expected_rows, base["rows"], strict=True):
        for key, value in expected.items():
            if observed[key] != value:
                raise AssertionError(f"Dataset drift: {key}")
    checks = {"analysis_exact_recomputation": True, "dataset_identity": True, "archive_identity": True}
    if (output / "interventions.json").exists():
        result = read(output / "interventions.json")
        recalculated = summarize_interventions(result["rows"], {r["example_id"]: r for r in base["rows"]},
            base["config"]["seed"]+30, base["config"]["bootstrap_samples"])
        if result["summary"] != recalculated:
            raise AssertionError("Intervention summary mismatch")
        checks["intervention_exact_recomputation"] = True
        checks["patched_generation_agreement"] = result["generation_agreement"]
    return checks


def verify_primary(output: Path) -> dict:
    """No GPU, Stage 2 archive, or all-layer Stage 3 archive required."""
    baseline = read(output / "baseline.json")
    saved = read(output / "analysis.json")
    archive = np.load(output / "selected_site.npz", allow_pickle=False)
    if archive["example_ids"].tolist() != [r["example_id"] for r in baseline["rows"]]:
        raise AssertionError("Primary archive identity mismatch")
    metrics, directions = evaluate_site(baseline["rows"], archive["activations"])
    expected = {k: v for k, v in saved["selected"].items() if k not in {"position", "layer"}}
    if metrics != expected or behavior_summary(baseline["rows"]) != saved["behavior"]:
        raise AssertionError("Primary metrics differ from raw recomputation")
    for name, direction in directions.items():
        np.testing.assert_array_equal(direction, archive[name])
    scores = archive["activations"] @ directions["auth_clean"]
    projection = projection_summary(baseline["rows"], scores, baseline["config"]["seed"]+10, baseline["config"]["bootstrap_samples"])
    if projection != saved["projection"]:
        raise AssertionError("Primary uncertainty does not reproduce")
    return {"primary_representation_exact": True, "primary_directions_exact": True,
            "behavior_exact": True, "primary_uncertainty_exact": True}


def manifest(output: Path) -> None:
    hashes = {p.relative_to(output).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"manifest.json"}}
    dump(output / "manifest.json", {"created_at_utc": now(), "files": hashes})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("screen", "formal", "analyze", "intervene", "verify", "verify-primary", "manifest"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_factorial.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    config = read(config_path)
    output = root / config["output_root"] / "formal-v1"
    if args.phase in {"screen", "formal"}:
        output = inference(config, config_path, args.phase, root)
        if args.phase == "formal":
            summary = analyze(output, root)
            print(json.dumps({"selected": {k: summary["selected"][k] for k in ("position", "layer", "selection_score")}, "gates": summary["gates"]}, indent=2), flush=True)
    elif args.phase == "analyze":
        summary = analyze(output, root)
        print(json.dumps({"gates": summary["gates"], "selected_layer": summary["selected"]["layer"]}, indent=2))
    elif args.phase == "intervene":
        result = intervene(output, root)
        if result.get("status") == "skipped":
            dump(output / "intervention_skipped.json", result)
        print(json.dumps(result.get("summary", result), indent=2), flush=True)
    elif args.phase == "verify":
        print(json.dumps(verify(output, root), indent=2))
    elif args.phase == "verify-primary":
        print(json.dumps(verify_primary(output), indent=2))
    elif args.phase == "manifest":
        manifest(output)


if __name__ == "__main__":
    main()
