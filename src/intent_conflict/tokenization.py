from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def prepare_messages(
    messages: list[dict[str, str]], system_role_policy: str = "native"
) -> list[dict[str, str]]:
    if system_role_policy == "native":
        return [dict(message) for message in messages]
    if system_role_policy != "merge_into_user":
        raise ValueError(f"Unknown system_role_policy: {system_role_policy}")
    if not messages or messages[0].get("role") != "system":
        return [dict(message) for message in messages]
    if len(messages) < 2 or messages[1].get("role") != "user":
        raise ValueError("merge_into_user requires system followed by user")
    merged_user = {
        "role": "user",
        "content": (
            "System instruction:\n"
            + messages[0]["content"]
            + "\n\nUser request:\n"
            + messages[1]["content"]
        ),
    }
    return [merged_user, *[dict(message) for message in messages[2:]]]


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool | None,
    system_role_policy: str = "native",
) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prepared = prepare_messages(messages, system_role_policy=system_role_policy)
    ids = tokenizer.apply_chat_template(prepared, **kwargs)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)
