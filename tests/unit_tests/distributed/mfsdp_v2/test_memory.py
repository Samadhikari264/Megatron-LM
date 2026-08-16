# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Memory-accounting tests for Megatron-FSDP."""

import logging

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


class MultiChildModel(nn.Module):
    """Model with direct parameters and multiple child FsdpModules."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class ElementwiseModel(nn.Module):
    """Small activation path over a large FSDP-managed weight."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the first weight row to an activation tensor."""
        return torch.relu(x + self.weight[0])


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MB"


def test_forward_peak_memory_bounds_in_flight_child_all_gathers(distributed_setup):
    """Forward peak memory should stay below three live child all-gathers."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=4).to(dtype=dtype, device=device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    resting_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize(device)
    peak_delta = torch.cuda.max_memory_allocated(device) - resting_allocated

    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()
    bound_nbytes = 3 * child_weight_nbytes

    # A parent forward should keep one previous child unsharded until its compute
    # stream consumer is safe, plus the current child being unsharded. The bound
    # is looser than two child weights to avoid coupling this test to CUDA
    # allocator granularity and small temporary buffers, while still catching
    # delayed releases piling up across the four child layers.
    assert peak_delta < bound_nbytes, (
        "FSDP forward peak memory exceeded the in-flight all-gather bound: "
        f"rank={rank}, peak_delta={_mb(peak_delta)}, "
        f"three_child_weights={_mb(bound_nbytes)}"
    )


def test_deleted_model_releases_fsdp_storage(distributed_setup):
    """Deleting an FSDP model should release its persistent storage."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    # Earlier tests may retain process-global CUDA allocations such as the
    # CuBLAS workspace. Capture them before creating this model, so the test
    # only detects storage retained by the deleted FSDP model itself.
    allocated_before = torch.cuda.memory_allocated(device)
    model = ElementwiseModel(dim=8192).to(dtype=torch.bfloat16, device=device)
    with fully_shard_context(device=device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.ones(1, 8192, dtype=torch.bfloat16, device=device)
    output = model(x)
    del output, x, model
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) - allocated_before < 1024**2


def test_root_forward_returns_to_resting_memory(distributed_setup):
    """Root forward should release child all-gather storage before returning."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=2).to(dtype=dtype, device=device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    resting_allocated = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        output = model(x)
    del output
    torch.cuda.synchronize(device)
    allocated_after_forward = torch.cuda.memory_allocated(device)
    extra_allocated = allocated_after_forward - resting_allocated
    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()

    assert extra_allocated < child_weight_nbytes, (
        "Root forward did not return to resting memory after draining child releases: "
        f"rank={rank}, extra_allocated={_mb(extra_allocated)}, "
        f"one_child_weight={_mb(child_weight_nbytes)}"
    )


def test_root_backward_returns_to_resting_memory(distributed_setup):
    """Root backward should release child all-gather storage before returning."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    dim = 4096
    dtype = torch.bfloat16
    model = MultiChildModel(dim=dim, num_children=2).to(dtype=dtype, device=device)
    placements = _flat_placements()
    policy = MixedPrecisionPolicy(main_params_dtype=dtype, main_grads_dtype=dtype)
    with fully_shard_context(device=device):
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, placements=placements, mixed_precision_policy=policy)
        fully_shard(model, mesh=mesh, placements=placements, mixed_precision_policy=policy)

    x = torch.randn(2, dim, device=device, dtype=dtype, requires_grad=True)
    output = model(x)
    loss = output.float().square().mean()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    allocated_before_backward = torch.cuda.memory_allocated(device)

    loss.backward()
    del loss, output
    torch.cuda.synchronize(device)
    allocated_after_backward = torch.cuda.memory_allocated(device)
    extra_allocated = allocated_after_backward - allocated_before_backward
    child_weight_nbytes = dim * dim * torch.empty((), dtype=dtype).element_size()

    assert extra_allocated < child_weight_nbytes, (
        "Root backward did not return to resting memory after draining child releases: "
        f"rank={rank}, extra_allocated={_mb(extra_allocated)}, "
        f"one_child_weight={_mb(child_weight_nbytes)}"
    )


def test_fully_shard_reduces_peak_training_memory(distributed_setup):
    """Per-layer FSDP should reduce peak CUDA memory during training."""
    rank = distributed_setup.rank
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")
    mesh = init_device_mesh(device.type, (world_size,))
    dim = 1024
    layers = 16
    batch = 8
    steps = 2
    dtype = torch.bfloat16

    def train_steps(model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor) -> None:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model(x).sum().backward()
            optimizer.step()

    torch.manual_seed(4321)
    baseline = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.01)
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(baseline, baseline_optimizer, x)
    torch.cuda.synchronize(device)
    baseline_peak = torch.cuda.max_memory_allocated(device)

    del baseline_optimizer
    del baseline
    del x
    torch.cuda.empty_cache()

    torch.manual_seed(4321)
    model = nn.Sequential(*[nn.Linear(dim, dim, dtype=dtype) for _ in range(layers)]).to(device)
    with fully_shard_context(device=device):
        for layer in model:
            fully_shard(
                layer,
                mesh=mesh,
                placements=_flat_placements(),
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_params_dtype=dtype, main_grads_dtype=dtype
                ),
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    torch.cuda.empty_cache()

    x = torch.randn(batch, dim, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(model, optimizer, x)
    torch.cuda.synchronize(device)
    sharded_peak = torch.cuda.max_memory_allocated(device)
    logger.info(
        "FSDP peak memory: rank=%s, baseline=%s, sharded=%s",
        rank,
        _mb(baseline_peak),
        _mb(sharded_peak),
    )

    assert sharded_peak < baseline_peak
