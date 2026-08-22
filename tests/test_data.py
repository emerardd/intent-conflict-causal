from intent_conflict.data import build_examples, paired_key, render_messages


def test_every_pair_holds_candidate_action_fixed() -> None:
    examples = build_examples(max_scenarios=3)
    grouped = {}
    for example in examples:
        grouped.setdefault(paired_key(example), []).append(example)
    assert grouped
    for pair in grouped.values():
        assert len(pair) == 2
        assert {item.authorized for item in pair} == {False, True}
        assert len({item.candidate_action_text for item in pair}) == 1
        assert {item.expected_label for item in pair} == {"A", "B"}


def test_each_prompt_contains_both_permission_and_prohibition() -> None:
    for example in build_examples(max_scenarios=2):
        constraint = example.constraint_text.lower()
        assert ("may" in constraint or "permitted" in constraint)
        assert ("do not" in constraint or "prohibited" in constraint)
        rendered = render_messages(example)[1]["content"]
        assert example.candidate_action_text in rendered


def test_candidate_position_is_counterbalanced() -> None:
    examples = build_examples(max_scenarios=1)
    positions = [example.candidate_position for example in examples]
    assert positions.count("candidate_first") == positions.count("distractor_first")


def test_stage2_has_forty_symmetric_scenarios() -> None:
    examples = build_examples(dataset_version="stage2_neutral")
    assert len(examples) == 40 * 4 * 2
    scenario_ids = {example.scenario_id for example in examples}
    assert len(scenario_ids) == 40
    for scenario_id in scenario_ids:
        scenario_examples = [item for item in examples if item.scenario_id == scenario_id]
        assert len(scenario_examples) == 8


def test_stage2_v2_uses_unique_unseen_codewords() -> None:
    examples = build_examples(dataset_version="stage2_neutral_v2")
    assert len(examples) == 40 * 4 * 2
    actions = {}
    for example in examples:
        actions.setdefault(example.scenario_id, example.candidate_action_text)
    assert len(set(actions.values())) == 40
    assert all('"alpha"' not in action and '"beta"' not in action for action in actions.values())
