from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

FALLBACK_CHAT_TEMPLATE = """{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>assistant\n' }}{% endif %}"""


def _get_attr_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def resolve_decoder_layers(model: Any) -> Any:
    candidates = [
        "model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        try:
            layers = _get_attr_path(model, path)
        except AttributeError:
            continue
        if len(layers) > 0:
            return layers
    raise AttributeError("Could not locate decoder layers: " + ", ".join(candidates))


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = hidden
        return output
    raise TypeError(f"Unsupported decoder layer output type: {type(output)!r}")


@dataclass
class PromptEvaluation:
    margin: float
    logp_execute: float
    logp_block: float
    predicted_label: str
    activations: np.ndarray


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    layers: Any
    execute_token_id: int
    block_token_id: int

    @classmethod
    def load(
        cls,
        model_name: str,
        model_loader: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        allow_fallback_chat_template: bool,
        quantization: str | None,
        tokenizer_loader: str = "auto",
        max_memory: dict[str, str] | None = None,
        offload_folder: str | None = None,
    ) -> "LoadedModel":
        torch_dtype = DTYPES[dtype]
        if tokenizer_loader == "auto":
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
        elif tokenizer_loader == "mistral_common":
            from transformers import MistralCommonBackend

            tokenizer = MistralCommonBackend.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown tokenizer_loader: {tokenizer_loader}")
        if tokenizer_loader == "auto" and tokenizer.chat_template is None:
            if not allow_fallback_chat_template:
                raise ValueError("Tokenizer has no native chat template")
            tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE

        if model_loader == "causal_lm":
            model_class = AutoModelForCausalLM
        elif model_loader == "multimodal_lm":
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise RuntimeError(
                    "AutoModelForMultimodalLM is required for this model"
                ) from exc
            model_class = AutoModelForMultimodalLM
        elif model_loader == "mistral3":
            from transformers import Mistral3ForConditionalGeneration

            model_class = Mistral3ForConditionalGeneration
        else:
            raise ValueError(f"Unknown model_loader: {model_loader}")

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": True,
            "dtype": torch_dtype,
        }
        unsharded_quantizations = (
            None,
            "none",
            "checkpoint_fp8",
            "checkpoint_fp8_dequant_bf16",
        )
        if quantization in unsharded_quantizations:
            if quantization == "checkpoint_fp8_dequant_bf16":
                from transformers import FineGrainedFP8Config

                if torch_dtype != torch.bfloat16:
                    raise ValueError(
                        "checkpoint_fp8_dequant_bf16 requires dtype=bfloat16"
                    )
                load_kwargs["quantization_config"] = FineGrainedFP8Config(
                    dequantize=True
                )
            if device == "auto":
                load_kwargs.update(
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                if max_memory:
                    load_kwargs["max_memory"] = {
                        int(key) if str(key).isdigit() else key: value
                        for key, value in max_memory.items()
                    }
                if offload_folder:
                    load_kwargs["offload_folder"] = offload_folder
                    load_kwargs["offload_state_dict"] = True
        elif quantization == "4bit_nf4":
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                ),
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        model = model_class.from_pretrained(model_name, **load_kwargs)
        if quantization in unsharded_quantizations and device != "auto":
            resolved_device = torch.device(device)
            model.to(resolved_device)
        else:
            resolved_device = model.get_input_embeddings().weight.device
        model.eval()

        execute_ids = tokenizer.encode("A", add_special_tokens=False)
        block_ids = tokenizer.encode("B", add_special_tokens=False)
        if len(execute_ids) != 1 or len(block_ids) != 1:
            raise ValueError(
                "Decision labels must each be one token; "
                f"got A={execute_ids}, B={block_ids}"
            )
        if execute_ids[0] == block_ids[0]:
            raise ValueError("A and B unexpectedly map to the same token")
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            layers=resolve_decoder_layers(model),
            execute_token_id=execute_ids[0],
            block_token_id=block_ids[0],
        )

    @contextmanager
    def activation_patch(
        self,
        layer_index: int,
        token_index: int,
        replacement: np.ndarray,
        expected_sequence_length: int,
    ) -> Iterator[None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] != expected_sequence_length:
                return output
            modified = hidden.clone()
            replacement_tensor = torch.as_tensor(
                replacement, device=modified.device, dtype=modified.dtype
            )
            modified[:, token_index, :] = replacement_tensor
            return _replace_hidden(output, modified)

        handle = self.layers[layer_index].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def evaluate_prompt(
        self,
        prompt_ids: list[int],
        layer_indices: list[int],
    ) -> PromptEvaluation:
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=tensor,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        log_probs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        logp_execute = float(log_probs[self.execute_token_id].detach().cpu())
        logp_block = float(log_probs[self.block_token_id].detach().cpu())
        activations = np.stack(
            [
                output.hidden_states[layer + 1][0, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
                for layer in layer_indices
            ]
        )
        margin = logp_block - logp_execute
        return PromptEvaluation(
            margin=margin,
            logp_execute=logp_execute,
            logp_block=logp_block,
            predicted_label="B" if margin >= 0 else "A",
            activations=activations,
        )

    def patched_margin(
        self,
        prompt_ids: list[int],
        layer_index: int,
        replacement: np.ndarray,
    ) -> float:
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with self.activation_patch(
            layer_index=layer_index,
            token_index=len(prompt_ids) - 1,
            replacement=replacement,
            expected_sequence_length=len(prompt_ids),
        ), torch.inference_mode():
            logits = self.model(
                input_ids=tensor,
                use_cache=False,
                return_dict=True,
            ).logits
        log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        return float(
            (log_probs[self.block_token_id] - log_probs[self.execute_token_id])
            .detach()
            .cpu()
        )
