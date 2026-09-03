"""Native-chat measurement with explicit positions and unconstrained greedy checks."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import inspect
import numpy as np
import torch
from .model import _replace_hidden


@contextmanager
def checked_patch(loaded, layer: int, position: int, vector: np.ndarray, length: int):
    calls = [0]
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != length:
            return output
        calls[0] += 1
        changed = hidden.clone()
        changed[:, position, :] = torch.as_tensor(vector, device=changed.device, dtype=changed.dtype)
        return _replace_hidden(output, changed)
    handle = loaded.layers[layer].register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()
    if calls[0] != 1:
        raise RuntimeError(f"Expected one patched prefill, got {calls[0]}")


def evaluate(loaded, ids: list[int], positions: dict[str, int], layers: list[int], patch=None):
    captures = {}
    handles = []
    if patch is None:
        for layer in layers:
            def capture(_module, _inputs, output, key=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[key] = hidden[0, list(positions.values())].detach().float().cpu().numpy()
            handles.append(loaded.layers[layer].register_forward_hook(capture))
    tensor = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    kwargs = {"input_ids": tensor, "use_cache": False, "return_dict": True}
    if "logits_to_keep" in inspect.signature(loaded.model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    context = checked_patch(loaded, patch[0], patch[1], patch[2], len(ids)) if patch else nullcontext()
    try:
        with context, torch.inference_mode():
            logits = loaded.model(**kwargs).logits[0, -1].float()
            logp = torch.log_softmax(logits, dim=-1)
            values, indices = torch.topk(logp, 5)
            aid, bid = loaded.execute_token_id, loaded.block_token_id
            la, lb = float(logp[aid]), float(logp[bid])
            top1 = int(indices[0])
            row = {"logp_A": la, "logp_B": lb, "margin": lb-la,
                   "ab_mass": float(torch.exp(logp[aid])+torch.exp(logp[bid])),
                   "ab_label": "B" if lb >= la else "A", "top1_id": top1,
                   "top1_label": "A" if top1 == aid else "B" if top1 == bid else "OTHER",
                   "top1_text": loaded.tokenizer.decode([top1]),
                   "top5": [{"id": int(i), "text": loaded.tokenizer.decode([int(i)]), "logp": float(v)} for i, v in zip(indices, values)]}
    finally:
        for handle in handles:
            handle.remove()
    hidden = np.stack([captures[layer] for layer in layers], axis=1) if captures else None
    return row, hidden


def generate(loaded, ids: list[int], max_new_tokens: int, patch=None) -> tuple[list[int], str]:
    tensor = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    context = checked_patch(loaded, patch[0], patch[1], patch[2], len(ids)) if patch else nullcontext()
    with context, torch.inference_mode():
        output = loaded.model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=loaded.tokenizer.eos_token_id)
    new_ids = output[0, len(ids):].tolist()
    return new_ids, loaded.tokenizer.decode(new_ids, skip_special_tokens=True)
