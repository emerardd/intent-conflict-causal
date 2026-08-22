from intent_conflict.tokenization import prepare_messages


def test_merge_system_role_preserves_both_messages() -> None:
    messages = [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    prepared = prepare_messages(messages, system_role_policy="merge_into_user")
    assert prepared == [
        {
            "role": "user",
            "content": "System instruction:\nSYSTEM TEXT\n\nUser request:\nUSER TEXT",
        }
    ]
    assert messages[0]["role"] == "system"
