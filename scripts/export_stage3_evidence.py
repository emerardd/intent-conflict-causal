"""Finalize local Stage 3 audit and integrity manifests; never uploads anything."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_conflict.stage3 import dump, manifest, now, sha
from intent_conflict.stage3_audit import audit_screen
from audit_stage3 import audit_formal


def main():
    base = ROOT / "results/stage3"
    dump(base / "audit.json", {
        "created_at_utc": now(),
        "v1": audit_screen(base / "screen-v1", True),
        "v2_pilot": audit_screen(base / "remapping-v2/screen-v1", True),
        "v2_formal": audit_formal(base / "remapping-v2/formal-v1"),
        "scope": "Independent artifact recomputation, not independent model-inference replication.",
    })
    files = [ROOT / "README.md"]
    for folder, pattern in (("docs", "stage3*.md"), ("configs", "stage3*.json"),
                            ("src/intent_conflict", "*.py"), ("tests", "test_*.py"), ("scripts", "*.py")):
        files += list((ROOT / folder).glob(pattern))
    dump(base / "delivery.json", {
        "created_at_utc": now(),
        "v1_status": "pilot_failed; formal_and_intervention_not_run",
        "v2_status": "pilot_formal_intervention_completed; causal_support_gate_failed",
        "no_training_no_new_model_download_no_git_commit_no_push": True,
        "claim": "Cross-mapping authorization-related decodability; no reliable effect of the tested one-dimensional intervention.",
        "remaining_confound": "authorization versus normative execute/block semantics",
        "current_delivery_source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)},
        "note": "Current delivery includes post-run audit/docs/tests. Actual inference sources are separately preserved in each run's source_snapshot.",
    })
    for run in (base / "screen-v1", base / "remapping-v2/screen-v1", base / "remapping-v2/formal-v1"):
        manifest(run)
    manifest(base)
    print(base / "audit.json")
    print(base / "delivery.json")
    print(base / "manifest.json")


if __name__ == "__main__":
    main()
