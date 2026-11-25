# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from packaging import version
from torch import Tensor

from verl.models.mcore.util import (
    build_vlm_attn_mask_bshd,
    build_vlm_attn_mask_thd,
    postprocess_bshd_engine,
    preprocess_bshd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output, postprocess_thd_engine

_FUSED_FORWARD_MODE_ATTR = "_verl_fused_forward_mode"
_HOOK_MODE = "hook"
# The model's ``forward`` does not take ``output_processor`` (e.g. Megatron-Bridge's Qwen3VLGPTModel, used by
# Qwen3.5), but its ``_postprocess`` does (Megatron Core >= 0.18). We wrap ``_postprocess`` and hand it the
# processor through an attribute set around the call, so the model's own forward (mRoPE, deepstack) still runs.
_POSTPROCESS_MODE = "postprocess"
_LEGACY_MODE = "legacy"
_PENDING_OUTPUT_PROCESSOR_ATTR = "_verl_pending_output_processor"


def _supports_output_processor_hook(patching_model: torch.nn.Module) -> bool:
    """Check whether the model supports Megatron's native output-processor hook.

    The ``output_processor`` / ``output_processor_context`` contract was
    introduced in Megatron Core 0.18.0. Models without this contract fall back
    to the legacy ``forward`` monkey patch.

    TODO: Remove this check and the legacy patch once all supported Megatron
    stacks require Megatron Core 0.18.0 or newer.
    """
    parameters = inspect.signature(patching_model.forward).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _supports_postprocess_hook(patching_model: torch.nn.Module) -> bool:
    """Check whether ``_postprocess`` takes the output-processor hook even though ``forward`` does not."""
    postprocess = getattr(patching_model, "_postprocess", None)
    if postprocess is None:
        return False
    parameters = inspect.signature(postprocess).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _resolve_fused_forward_mode(patching_model: torch.nn.Module) -> str:
    if _supports_output_processor_hook(patching_model):
        return _HOOK_MODE
    if _supports_postprocess_hook(patching_model):
        return _POSTPROCESS_MODE
    return _LEGACY_MODE


def _get_fused_forward_mode(model: torch.nn.Module) -> str:
    model = unwrap_model(model)
    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None and hasattr(model, "language_model"):
        mode = getattr(model.language_model, _FUSED_FORWARD_MODE_ATTR, None)
    return mode if mode in (_HOOK_MODE, _POSTPROCESS_MODE, _LEGACY_MODE) else _LEGACY_MODE


def _use_output_processor_hook(model: torch.nn.Module) -> bool:
    return _get_fused_forward_mode(model) in (_HOOK_MODE, _POSTPROCESS_MODE)


def _call_with_output_processor(model: torch.nn.Module, forward_kwargs: dict, temperature: float):
    """Run ``model`` with ``fused_output_processor`` at its postprocess boundary (hook or postprocess mode)."""
    context = FusedOutputProcessorContext(temperature=temperature)
    if _get_fused_forward_mode(model) == _HOOK_MODE:
        return model(**forward_kwargs, output_processor=fused_output_processor, output_processor_context=context)
    patched = _get_patching_model(model)
    setattr(patched, _PENDING_OUTPUT_PROCESSOR_ATTR, (fused_output_processor, context))
    try:
        return model(**forward_kwargs)
    finally:
        setattr(patched, _PENDING_OUTPUT_PROCESSOR_ATTR, None)


@dataclass
class FusedOutputProcessorContext:
    """Context passed through Megatron's native output-processor hook."""

    temperature: float


def fused_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    context,
    config,
    **_ignored,
):
    """Compute fused log probabilities and entropy at Megatron's postprocess boundary."""
    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Megatron passes the shared embedding as output_weight for tied models. For
    # untied models the weight lives on output_layer.
    weight = output_weight if output_weight is not None else output_layer.weight

    temperature = context.temperature
    # Megatron's hidden states are sequence-first [s, b, h] while labels are batch-first [b, s]. The kernel flattens
    # both, so with b > 1 (padded bshd micro-batches) the labels must be put in the same [s, b] order first, and the
    # results turned back to [b, s]. With b == 1 (thd packing, or bshd with one sequence) both orders agree and the
    # outputs keep their original flat shape.
    batch_first = (
        hidden_states.dim() == 3
        and labels is not None
        and labels.dim() == 2
        and hidden_states.shape[1] > 1
        and tuple(labels.shape) == (hidden_states.shape[1], hidden_states.shape[0])
    )
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        weight,
        labels.transpose(0, 1).contiguous() if batch_first else labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )
    if batch_first:
        seq_len, batch_size = hidden_states.shape[:2]
        logprobs = logprobs.view(seq_len, batch_size).transpose(0, 1)
        entropy = entropy.view(seq_len, batch_size).transpose(0, 1)

    if has_config_logger_enabled(config):
        payload = OrderedDict(
            {
                "input_ids": _ignored.get("input_ids"),
                "position_ids": _ignored.get("position_ids"),
                "attention_mask": _ignored.get("attention_mask"),
                "decoder_input": _ignored.get("decoder_input"),
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs
    return output


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is None:
        return

    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None:
        mode = _resolve_fused_forward_mode(model)
        setattr(model, _FUSED_FORWARD_MODE_ATTR, mode)

    if mode == _HOOK_MODE:
        return

    if mode == _POSTPROCESS_MODE:
        if not hasattr(model, "postprocess_backup"):
            model.postprocess_backup = model._postprocess
            setattr(model, _PENDING_OUTPUT_PROCESSOR_ATTR, None)

            def _postprocess_with_pending_processor(*args, **kwargs):
                pending = getattr(model, _PENDING_OUTPUT_PROCESSOR_ATTR, None)
                if pending is not None and kwargs.get("output_processor") is None:
                    kwargs["output_processor"], kwargs["output_processor_context"] = pending
                return model.postprocess_backup(*args, **kwargs)

            model._postprocess = _postprocess_with_pending_processor
        return

    assert version.parse(mcore.__version__) >= version.parse("0.13.0"), (
        "Fused forward patching requires mecore >= 0.13.0"
    )
    if not hasattr(model, "forward_backup"):
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is None or _get_fused_forward_mode(model) == _HOOK_MODE:
        return
    if _get_fused_forward_mode(model) == _POSTPROCESS_MODE:
        if hasattr(model, "postprocess_backup"):
            model._postprocess = model.postprocess_backup
            delattr(model, "postprocess_backup")
        return
    if hasattr(model, "forward_backup"):
        model.forward = model.forward_backup
        delattr(model, "forward_backup")


def fused_forward_model_gen(vision_model: bool = False):
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process: bool = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            # workaround for supporting sequence packing with context parallelism
            # cp split with sequence packing will make model lose vision token information, so we need to keep
            # the original input_ids and pack them after vision embedding is calculated,
            # cooporate with mbridge
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        if _use_output_processor_hook(model):
            input_args.pop("temperature", None)
            output_orig: CausalLMOutputForPPO = _call_with_output_processor(model, input_args, temperature)
        else:
            output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def fused_forward_model_engine(vision_model: bool = False):
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        cp_layout: str = "zigzag",
        local_cp_size: int | None = None,
        router_padding_mask: Tensor | None = None,
        pad_to_length_bucket: int | None = None,
    ):
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        config = unwrap_model(model).config
        min_local_rows = (
            config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
        )

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids,
            pre_process=pre_process,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if router_padding_mask is not None:
            model_kwargs["padding_mask"] = router_padding_mask
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        attention_mask = None
        if vision_model:
            input_ids_rmpad, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        labels_rmpad = labels_rmpad.contiguous()
        forward_kwargs = dict(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            **model_kwargs,
        )
        if _use_output_processor_hook(model):
            output_orig: CausalLMOutputForPPO = _call_with_output_processor(model, forward_kwargs, temperature)
        else:
            output_orig: CausalLMOutputForPPO = model(temperature=temperature, **forward_kwargs)

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs,
            packed_seq_params,
            input_ids,
            input_ids.shape[0],
            post_process=post_process,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy,
                packed_seq_params,
                input_ids,
                input_ids.shape[0],
                post_process=post_process,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
            )
            output["entropy"] = entropy

        return output

    return fused_forward_model_engine_inner


def fused_forward_model_engine_bshd(vision_model: bool = False):
    """Fused log-probs for padded (bshd) inputs, for models that cannot use thd packing.

    Megatron-LM does not support packed sequences for Qwen3.5's gated delta-net layers yet, so those models run
    with ``use_remove_padding=False``. The unfused bshd path then materializes the full ``[b, s, vocab]`` logits
    (248k vocab for Qwen3.5: ~16 GB at 11k tokens with the fp32 copy). This function mirrors the bshd branch of
    ``gptmodel_forward_model_engine`` but asks the model to return log-probs (and entropy) computed by the chunked
    ``linear_cross_entropy`` kernel at the postprocess boundary, so the logits are never materialized.
    Requires the output-processor hook (hook or postprocess mode); the legacy forward patch is thd-only.
    """

    def fused_forward_model_engine_bshd_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        cp_layout: str = "zigzag",
        local_cp_size: int | None = None,
        router_padding_mask: Tensor | None = None,
        pad_to_length_bucket: int | None = None,
        forced_max_seqlen: int | None = None,
    ):
        assert local_cp_size is None, "dynamic CP is not supported for bshd format"
        assert router_padding_mask is None, "router padding mask is not supported for fused bshd"
        assert pad_to_length_bucket is None, "pad_to_length is not supported for bshd format"
        if not _use_output_processor_hook(model):
            raise NotImplementedError(
                "fused kernels with use_remove_padding=False need the Megatron output-processor hook "
                "(Megatron Core >= 0.18); the legacy forward patch only supports thd"
            )
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process
        use_fp8_padding = unwrap_model(model).config.fp8 in ["e4m3", "hybrid"]
        batch_size = input_ids.shape[0]

        input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd_engine(
            input_ids, pre_process=pre_process, use_fp8_padding=use_fp8_padding, forced_max_seqlen=forced_max_seqlen
        )
        labels_bshd = preprocess_bshd_engine(
            labels, pre_process=True, need_roll=True, use_fp8_padding=use_fp8_padding, forced_max_seqlen=forced_max_seqlen
        )[0].contiguous()

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        # Same as the unfused bshd branch: VLMs take bshd input_ids plus their own attention mask.
        if vision_model:
            input_ids_bshd, attention_mask = build_vlm_attn_mask_bshd(
                input_ids, batch_size, pad_token_id, forced_max_seqlen=forced_max_seqlen
            )
        else:
            attention_mask = attention_mask_bshd

        forward_kwargs = dict(
            input_ids=input_ids_bshd,
            attention_mask=attention_mask,
            position_ids=None if vision_model else position_ids_bshd,
            labels=labels_bshd,
            **model_kwargs,
        )
        output_orig: CausalLMOutputForPPO = _call_with_output_processor(model, forward_kwargs, temperature)
        if not post_process:
            return output_orig

        seq_len = labels_bshd.shape[1]
        log_probs = output_orig.log_probs.reshape(batch_size, seq_len)
        output = {"log_probs": postprocess_bshd_engine(log_probs, attention_mask_bshd, post_process=post_process)}
        if calculate_entropy:
            entropy = output_orig.entropy.reshape(batch_size, seq_len)
            output["entropy"] = postprocess_bshd_engine(entropy, attention_mask_bshd, post_process=post_process)
        return output

    return fused_forward_model_engine_bshd_inner


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    padding_mask: Tensor | None = None,
    **kwargs,
) -> CausalLMOutputForPPO:
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preprocess_kwargs = {}
    if padding_mask is not None:
        # Only forward the kwarg when set: older Megatron-Core _preprocess
        # signatures (without MoE router padding support) stay compatible.
        preprocess_kwargs["padding_mask"] = padding_mask
    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
        **preprocess_kwargs,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    decoder_extra_block_kwargs = extra_block_kwargs or {}
    if padding_mask is not None:
        # _preprocess scatters the mask across sequence-parallel ranks.
        decoder_extra_block_kwargs["padding_mask"] = preproc_output[5]
    if getattr(model.config, "moe_n_hash_layers", 0) > 0 and input_ids is not None:
        decoder_extra_block_kwargs["input_ids"] = input_ids

    # Run decoder.
    decoder_output = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **decoder_extra_block_kwargs,
        **kwargs,
    )
    hidden_states = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Get the output weight - use embedding weight if output_layer is None or weight is shared
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        # When embeddings are tied, use the embedding weight
        output_weight = model.embedding.word_embeddings.weight

    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        output_weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
