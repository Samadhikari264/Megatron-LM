# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import warnings
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
)
from megatron.core.utils import (
    flatten_batch_for_packed_sequences,
    is_te_min_version,
)
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import init_basic_mock_args
from tests.unit_tests.test_utilities import Utils

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# Match the standard unit-test wrapper's low-memory NCCL settings. This test
# repeatedly creates model-parallel communicators and otherwise can fail in
# NCCL/NVLS setup before reaching the parity assertions.
os.environ.update({"NCCL_MAX_NCHANNELS": "1", "NCCL_NVLS_ENABLE": "0"})


_PARALLEL_EP_SIZE = 4
_SEQUENCE_LENGTH = 4096
_MICRO_BATCH_SIZE = 1
_VOCAB_SIZE = 8192
_SEED = 1234
_DIAGNOSTIC_REPEATS = 5
_NUM_LAYERS = 3
_LINEAR_ATTENTION_PATTERN = [1, 0, 0]
_LM_LOSS_ATOL = 0.003
_MTP_LOSS_ATOL = 0.003
_GRAD_NORM_ATOL = 0.1
_GRAD_NORM_RTOL = 0.01
_REFERENCE_LINEAR_CP_MODE = "headwise"
_REFERENCE_CP_PARTITION_MODE = "zigzag"
_CANDIDATE_LINEAR_CP_MODE = "chunkwise"
_CANDIDATE_CP_PARTITION_MODE = "contiguous"
_CP_LAYOUT_WARNING_SUBSTRINGS = (
    "missing precomputed context-parallel layout routes",
    "TransformerConfig.cp_partition_mode is deprecated and ignored",
)
_PARALLEL_CASES = (
    pytest.param(4, 1, _PARALLEL_EP_SIZE, False, id="cp4_tp1_ep4"),
)
_FULL_RECOMPUTE_CASES = (
    pytest.param(False, id="no_recompute"),
    pytest.param(True, id="full_recompute"),
)


def _destroy_model_parallel_without_barrier():
    if not Utils.inited:
        return
    torch.cuda.synchronize()
    parallel_state.destroy_model_parallel()
    Utils.inited = False
    torch.cuda.memory.empty_cache()


def _collect_parameter_state(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def _copy_state_to_model(source_state, model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            source = source_state[name]
            if source.shape == param.shape:
                param.copy_(source.to(device=param.device, dtype=param.dtype))
                continue

            raise AssertionError(
                f"Cannot copy parameter {name}: source shape {tuple(source.shape)}, "
                f"target shape {tuple(param.shape)}"
            )


def _assert_no_cp_layout_warnings(caught_warnings):
    for caught_warning in caught_warnings:
        message = str(caught_warning.message)
        assert not any(
            warning_substring in message
            for warning_substring in _CP_LAYOUT_WARNING_SUBSTRINGS
        ), message


def _make_config(
    context_parallel_size,
    tensor_model_parallel_size,
    expert_model_parallel_size,
    sequence_parallel,
    qkv_format,
    linear_cp_mode,
    cp_partition_mode,
    full_recompute,
):
    packed_kwargs = {}
    if qkv_format == "thd":
        packed_kwargs = {
            "sequence_packing_scheduler": "dp_balanced",
            "pad_packed_seq_alignment": "max",
            "max_seqlen_per_dp_cp_rank": _SEQUENCE_LENGTH,
        }

    recompute_kwargs = (
        {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        }
        if full_recompute
        else {
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    )

    return TransformerConfig(
        hidden_size=512,
        ffn_hidden_size=1024,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=_NUM_LAYERS,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        kv_channels=64,
        num_query_groups=2,
        qk_layernorm=True,
        attention_output_gate=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=_LINEAR_ATTENTION_PATTERN,
        linear_cp_mode=linear_cp_mode,
        cp_partition_mode=cp_partition_mode,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=tensor_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        calculate_per_token_loss=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_moe_experts=32,
        moe_layer_freq=1,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_gate=True,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=4,
        moe_grouped_gemm=True,
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_flex_dispatcher_num_sms=32,
        moe_permute_fusion=True,
        moe_router_fusion=True,
        moe_router_dtype="fp32",
        mtp_num_layers=1,
        mtp_loss_scaling_factor=1.0,
        mtp_use_repeated_layer=False,
        gdn_pre_gated_delta_rule_fusion=False,
        **packed_kwargs,
        **recompute_kwargs,
    )


def _initialize_gpt_model(
    config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
):
    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=config, vp_stage=vp_stage, pp_rank=0
    )
    mtp_block_spec = None
    if config.mtp_num_layers:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=transformer_layer_spec,
            use_transformer_engine=True,
            vp_stage=vp_stage,
            pp_rank=0,
        )

    model_kwargs = {
        "config": config,
        "transformer_layer_spec": transformer_layer_spec,
        "mtp_block_spec": mtp_block_spec,
        "vocab_size": _VOCAB_SIZE,
        "max_sequence_length": _SEQUENCE_LENGTH,
        "pre_process": pre_process,
        "post_process": post_process,
        "position_embedding_type": "rope",
        "rotary_percent": 0.25,
        "rotary_base": 10000000,
        "pg_collection": pg_collection,
        "vp_stage": vp_stage,
    }
    return GPTModel(**model_kwargs)


def _build_gpt_model(config, device):
    model = _initialize_gpt_model(config)
    model.to(device=device)
    return model


def _set_mock_args(args, config, context_parallel_size):
    init_basic_mock_args(args, config.tensor_model_parallel_size, 1, bf16=True)
    args.context_parallel_size = context_parallel_size
    args.cp_comm_type = "a2a" if context_parallel_size == 1 else "p2p"
    args.expert_model_parallel_size = config.expert_model_parallel_size
    args.expert_tensor_parallel_size = 1
    args.num_experts = config.num_moe_experts
    args.moe_ffn_hidden_size = config.moe_ffn_hidden_size
    args.moe_shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
    args.moe_shared_expert_gate = config.moe_shared_expert_gate
    args.moe_router_load_balancing_type = config.moe_router_load_balancing_type
    args.moe_router_topk = config.moe_router_topk
    args.moe_grouped_gemm = config.moe_grouped_gemm
    args.moe_aux_loss_coeff = config.moe_aux_loss_coeff
    args.moe_token_dispatcher_type = config.moe_token_dispatcher_type
    args.moe_flex_dispatcher_backend = config.moe_flex_dispatcher_backend
    args.moe_flex_dispatcher_num_sms = config.moe_flex_dispatcher_num_sms
    args.moe_permute_fusion = config.moe_permute_fusion
    args.moe_router_fusion = config.moe_router_fusion
    args.moe_router_dtype = config.moe_router_dtype
    args.mtp_num_layers = config.mtp_num_layers
    args.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
    args.mtp_use_repeated_layer = config.mtp_use_repeated_layer
    args.recompute_granularity = config.recompute_granularity
    args.recompute_method = config.recompute_method
    args.recompute_num_layers = config.recompute_num_layers
    args.linear_cp_mode = config.linear_cp_mode
    args.cp_partition_mode = config.cp_partition_mode
    args.sequence_parallel = config.sequence_parallel
    args.seq_length = _SEQUENCE_LENGTH
    args.max_position_embeddings = _SEQUENCE_LENGTH
    args.padded_vocab_size = _VOCAB_SIZE
    args.untie_embeddings_and_output_weights = True


def _make_thd_batch(device):
    padded_seq_lengths = [1024, 768, 1280, 1024]
    seq_lengths = [901, 629, 1103, 877]
    prompt_lengths = [0, 63, 257, 15]
    assert sum(padded_seq_lengths) == _SEQUENCE_LENGTH

    tokens = torch.zeros((_MICRO_BATCH_SIZE, _SEQUENCE_LENGTH), device=device, dtype=torch.long)
    labels = torch.zeros_like(tokens)
    loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
    padding_mask = torch.ones_like(tokens, dtype=torch.bool)
    position_ids = torch.empty_like(tokens)

    padded_offset = 0
    cu_seqlens = [0]
    cu_seqlens_padded = [0]
    for seq_length, padded_seq_length, prompt_length in zip(
        seq_lengths, padded_seq_lengths, prompt_lengths
    ):
        valid_end = padded_offset + seq_length
        padded_end = padded_offset + padded_seq_length
        seq_tokens = torch.randint(
            low=0,
            high=_VOCAB_SIZE,
            size=(_MICRO_BATCH_SIZE, seq_length),
            device=device,
            dtype=torch.long,
        )
        tokens[:, padded_offset:valid_end] = seq_tokens
        labels[:, padded_offset:valid_end] = (seq_tokens + 1) % _VOCAB_SIZE
        loss_mask[:, padded_offset + prompt_length : valid_end] = 1.0
        padding_mask[:, padded_offset:valid_end] = False
        position_ids[:, padded_offset:valid_end] = torch.arange(
            seq_length, device=device, dtype=torch.long
        )
        position_ids[:, valid_end:padded_end] = 0
        cu_seqlens.append(cu_seqlens[-1] + seq_length)
        cu_seqlens_padded.append(padded_end)
        padded_offset = padded_end

    cu_seqlens = torch.tensor([cu_seqlens], device=device, dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([cu_seqlens_padded], device=device, dtype=torch.int32)
    max_seqlen = torch.tensor([max(padded_seq_lengths)], device=device, dtype=torch.int32)
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": None,
        "padding_mask": padding_mask,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "max_seqlen": max_seqlen,
    }


def _prepare_batch_for_model(batch, cp_group, cp_partition_mode):
    batch = deepcopy(batch)
    batch = flatten_batch_for_packed_sequences(batch)

    for key in ("tokens", "labels", "loss_mask", "position_ids", "padding_mask"):
        if batch.get(key) is not None and batch[key].dim() == 2:
            batch[key] = batch[key].squeeze(0)

    cu_seqlens = batch["cu_seqlens"].squeeze(0)
    cu_seqlens_padded = batch["cu_seqlens_padded"].squeeze(0)
    max_seqlen = batch["max_seqlen"].squeeze(0)
    batch["cu_seqlens"] = cu_seqlens
    batch["cu_seqlens_padded"] = cu_seqlens_padded
    batch["max_seqlen"] = max_seqlen

    get_cp_slice_for_thd(
        batch,
        cp_group,
        keys=("tokens", "position_ids", "labels", "loss_mask", "padding_mask"),
        cp_partition_mode=cp_partition_mode,
    )
    for key in ("tokens", "labels", "loss_mask", "position_ids", "padding_mask"):
        if batch.get(key) is not None:
            batch[key] = batch[key].view(1, -1)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen.item()),
        max_seqlen_kv=int(max_seqlen.item()),
        cp_group=cp_group,
        cp_partition_mode=cp_partition_mode,
        total_tokens=int(cu_seqlens_padded[-1].item()),
        tokens_per_sample=_SEQUENCE_LENGTH,
        pad_between_seqs=True,
    )
    prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)
    return batch, packed_seq_params


def _global_grad_norm(model):
    grads = [
        param.grad.detach()
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    return torch.tensor(
        get_grad_norm_fp32(grads, grad_stats_parallel_group=torch.distributed.group.WORLD),
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    )


def _get_mtp_losses_from_tracker():
    MTPLossLoggingHelper.reduce_loss_in_tracker()
    tracker = MTPLossLoggingHelper.tracker
    assert "values" in tracker, "MTP loss tracker did not record any loss values."
    return tracker["values"].detach().float().clone()


def _loss_and_grad_stats(model, batch, packed_seq_params):
    model.zero_grad(set_to_none=True)
    MTPLossLoggingHelper.clean_loss_in_tracker()
    MTPLossAutoScaler.set_loss_scale(torch.ones((), device=torch.cuda.current_device()))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        loss = model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            packed_seq_params=packed_seq_params,
            padding_mask=batch.get("padding_mask"),
        )
        mtp_losses = _get_mtp_losses_from_tracker()
        numerator = (loss.float() * batch["loss_mask"]).sum()
        denominator = batch["loss_mask"].sum()
        (numerator / denominator.clamp(min=1)).backward()

    _assert_no_cp_layout_warnings(caught_warnings)
    grad_norm = _global_grad_norm(model)
    return numerator.detach(), denominator.detach(), mtp_losses.detach(), grad_norm.detach()


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not is_te_min_version("1.11.0"), reason="MoE grouped GEMM requires TE >= 1.11.")
@pytest.mark.parametrize("repeat_index", range(_DIAGNOSTIC_REPEATS))
@pytest.mark.parametrize("full_recompute", _FULL_RECOMPUTE_CASES)
@pytest.mark.parametrize(
    (
        "context_parallel_size,tensor_model_parallel_size,expert_model_parallel_size,"
        "sequence_parallel"
    ),
    _PARALLEL_CASES,
)
def test_qwen35_proxy_gdn_moe_chunkwise_loss_and_grad_matches_headwise(
    context_parallel_size,
    tensor_model_parallel_size,
    expert_model_parallel_size,
    sequence_parallel,
    full_recompute,
    repeat_index,
):
    min_world_size = max(
        context_parallel_size * tensor_model_parallel_size, expert_model_parallel_size
    )
    if not torch.cuda.is_available() or Utils.world_size < min_world_size:
        pytest.skip(f"GDN/MoE CP loss parity needs at least {min_world_size} CUDA ranks.")

    mock_args = parse_args(ignore_unknown_args=True)
    set_args(mock_args)

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=1,
            context_parallel_size=context_parallel_size,
        )
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        seed = _SEED + repeat_index
        torch.manual_seed(seed)
        batch = _make_thd_batch(device)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        reference_config = _make_config(
            context_parallel_size=context_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=sequence_parallel,
            qkv_format="thd",
            linear_cp_mode=_REFERENCE_LINEAR_CP_MODE,
            cp_partition_mode=_REFERENCE_CP_PARTITION_MODE,
            full_recompute=full_recompute,
        )
        _set_mock_args(mock_args, reference_config, context_parallel_size=context_parallel_size)

        reference_model = _build_gpt_model(reference_config, device)
        reference_model.train()

        cp_group = parallel_state.get_context_parallel_group()
        reference_batch, reference_packed_seq_params = _prepare_batch_for_model(
            batch,
            cp_group=cp_group,
            cp_partition_mode=reference_config.cp_partition_mode,
        )
        reference_num, reference_den, reference_mtp_losses, reference_grad_norm = (
            _loss_and_grad_stats(reference_model, reference_batch, reference_packed_seq_params)
        )
        reference_stats = torch.stack([reference_num.detach(), reference_den.detach()])
        torch.distributed.all_reduce(reference_stats, group=cp_group)
        reference_avg = (reference_stats[0] / reference_stats[1].clamp(min=1)).float().cpu()
        reference_mtp_losses = reference_mtp_losses.float().cpu()
        reference_grad_norm = reference_grad_norm.float().cpu()
        source_state = _collect_parameter_state(reference_model)

        torch.cuda.synchronize()
        del (
            reference_model,
            reference_batch,
            reference_packed_seq_params,
            reference_num,
            reference_den,
            reference_stats,
        )
        gc.collect()
        torch.cuda.empty_cache()

        model_parallel_cuda_manual_seed(seed)
        candidate_config = _make_config(
            context_parallel_size=context_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=sequence_parallel,
            qkv_format="thd",
            linear_cp_mode=_CANDIDATE_LINEAR_CP_MODE,
            cp_partition_mode=_CANDIDATE_CP_PARTITION_MODE,
            full_recompute=full_recompute,
        )
        _set_mock_args(mock_args, candidate_config, context_parallel_size=context_parallel_size)
        candidate_model = _build_gpt_model(candidate_config, device)
        candidate_model.train()
        _copy_state_to_model(source_state, candidate_model)

        candidate_batch, candidate_packed_seq_params = _prepare_batch_for_model(
            batch,
            cp_group=cp_group,
            cp_partition_mode=candidate_config.cp_partition_mode,
        )
        candidate_num, candidate_den, candidate_mtp_losses, candidate_grad_norm = (
            _loss_and_grad_stats(candidate_model, candidate_batch, candidate_packed_seq_params)
        )
        stats = torch.stack([candidate_num.detach(), candidate_den.detach()])
        torch.distributed.all_reduce(stats, group=cp_group)
        candidate_avg = stats[0] / stats[1].clamp(min=1)
        candidate_avg = candidate_avg.float().cpu()
        candidate_mtp_losses = candidate_mtp_losses.float().cpu()
        candidate_grad_norm = candidate_grad_norm.float().cpu()
        lm_loss_diff = (candidate_avg - reference_avg).abs()
        mtp_loss_diff = (candidate_mtp_losses - reference_mtp_losses).abs().max()
        grad_norm_diff = (candidate_grad_norm - reference_grad_norm).abs()

        if torch.distributed.get_rank() == 0:
            print(
                "GDN MoE CP loss/grad parity: "
                f"case=cp{context_parallel_size}_tp{tensor_model_parallel_size}_"
                f"ep{expert_model_parallel_size}"
                f"{'_sp' if sequence_parallel else ''} "
                f"layers={_NUM_LAYERS} linear_attention_pattern={_LINEAR_ATTENTION_PATTERN} "
                f"reference_cp_partition_mode={reference_config.cp_partition_mode} "
                f"candidate_cp_partition_mode={candidate_config.cp_partition_mode} "
                f"format=thd full_recompute={full_recompute} "
                f"repeat={repeat_index} seed={seed} "
                f"lm_reference={reference_avg.item():.8f} "
                f"lm_candidate={candidate_avg.item():.8f} "
                f"lm_diff={lm_loss_diff.item():.8f} "
                f"mtp_reference={reference_mtp_losses[0].item():.8f} "
                f"mtp_candidate={candidate_mtp_losses[0].item():.8f} "
                f"mtp_diff={mtp_loss_diff.item():.8f} "
                f"grad_norm_reference={reference_grad_norm.item():.8f} "
                f"grad_norm_candidate={candidate_grad_norm.item():.8f} "
                f"grad_norm_diff={grad_norm_diff.item():.8f}",
                flush=True,
            )

        torch.testing.assert_close(
            candidate_avg,
            reference_avg,
            atol=_LM_LOSS_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            candidate_mtp_losses,
            reference_mtp_losses,
            atol=_MTP_LOSS_ATOL,
            rtol=0.0,
        )
        torch.testing.assert_close(
            candidate_grad_norm,
            reference_grad_norm,
            atol=_GRAD_NORM_ATOL,
            rtol=_GRAD_NORM_RTOL,
        )
    finally:
        MTPLossLoggingHelper.clean_loss_in_tracker()
        _destroy_model_parallel_without_barrier()
