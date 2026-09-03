"""Post-run audit/guard regressions; not used for model selection or training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from intent_conflict.stage3 import dump, inference, intervene


def audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_stage3.py"
    spec = importlib.util.spec_from_file_location("standalone_stage3_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_pairwise_auc_includes_half_ties():
    audit = audit_module()
    y = np.array([False, False, True, True])
    assert audit.pairwise_auc(y, np.array([0, 1, 1, 2])) == 0.875
    assert audit.pairwise_auc(y, np.ones(4)) == 0.5
    assert audit.pairwise_auc(y, np.array([3, 2, 1, 0])) == 0


def test_independent_bootstrap_is_seeded_and_cluster_level():
    audit = audit_module()
    config = {"seed": 2, "bootstrap_samples": 100}
    np.testing.assert_array_equal(audit.ci([3., 3., 3.], config, 10), [3, 3])
    np.testing.assert_array_equal(audit.ci([1., 2., 5.], config, 10), audit.ci([1., 2., 5.], config, 10))


def test_failed_screen_forbids_formal_before_any_model_load(tmp_path):
    config = {"output_root": "results/example"}
    dump(tmp_path / "results/example/screen-v1/screen_summary.json", {"passed": False})
    with pytest.raises(RuntimeError, match="Pilot failed"):
        inference(config, tmp_path / "nonexistent-config.json", "formal", tmp_path)
    assert not (tmp_path / "results/example/formal-v1").exists()


def test_failed_eligibility_skips_intervention_before_model_load(tmp_path):
    dump(tmp_path / "analysis.json", {"intervention_eligible": False, "gates": {"behavior": False}})
    result = intervene(tmp_path, tmp_path)
    assert result["status"] == "skipped"
    assert not (tmp_path / "interventions.json").exists()


def test_completed_run_is_never_overwritten(tmp_path):
    (tmp_path / "screen-v1").mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        inference({"output_root": "."}, tmp_path / "config.json", "screen", tmp_path)
