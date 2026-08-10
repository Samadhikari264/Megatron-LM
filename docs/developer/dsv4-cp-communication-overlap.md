<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# DSv4 CP communication overlap

This note covers the ratio-4 indexer path in packed-THD context parallelism.
The boundary-token P2P exchange is independent of the two collectives below.

## Forward

The CP group gathers the two local compressor outputs in a fixed order:

```text
compute: indexer-K compressor | attention-KV compressor + local Q/weight projections
comm:                          | AG(indexer K)

compute: score-buffer init | indexer score + top-k | sparse attention
comm:                       | AG(attention KV)
```

`AG(indexer K)` is launched after the Indexer-K compressor and waited on at
the first sequence-major K consumer before top-k. The attention-KV compressor
and local indexer projections are independent and run between the launch and
wait.

`AG(attention KV)` is launched after the indexer score buffer is initialized
and waited on after top-k, before the gathered KV is concatenated for sparse
attention. The exact launch boundary requires cuDNN Frontend's
`post_output_init_callback` on `indexer_forward_wrapper`. Older wrappers fall
back to launching after score computation and can overlap only radix top-k.

## Backward

The fused indexer-loss forward saves the Indexer Q/K/weight gradients. At
backward entry, the global Indexer-K gradient is ready before sparse-attention
backward produces the global compressed-KV gradient.

```text
compute: sparse-attention bwd | local Q/weight bwd | attention-KV compressor bwd | Indexer-K compressor bwd
comm:                         | RS(attention KV) -> RS(indexer K)
wait:                                               ^ KV consumer                 ^ Indexer-K consumer
```

The fused autograd function first completes sparse-attention backward. It then
launches `RS(attention KV)` followed by `RS(indexer K)`. This avoids keeping an
early-arriving NCCL kernel resident throughout the long sparse-attention kernel.
Each wait is attached only to the corresponding local compressor gradient edge.
The newer local Q/weight branch runs first, the attention-KV consumer runs next,
and the Indexer-K consumer runs last. The second reduction can therefore remain
in flight while the attention-KV compressor backward executes.

The gathered forward tensors are detached from the generic gather backward so
each gradient is reduced exactly once.

All ranks enqueue collectives in the same order: Indexer-K before attention-KV
in forward, and attention-KV before Indexer-K in backward. The unfused and
zero-indexer-loss paths retain the standard synchronous backward mappings.

The implementation uses asynchronous process-group work handles and waits at
the first tensor consumer. It does not create application-owned CUDA streams,
so the ordering remains compatible with CUDA Graph capture.

## Profiling ranges

The implementation emits the following NVTX ranges:

- `dsv4_cp_indexer_k_all_gather_launch` and `dsv4_cp_indexer_k_all_gather_wait`
- `dsv4_cp_attention_kv_all_gather_launch` and `dsv4_cp_attention_kv_all_gather_wait`
- `dsv4_cp_indexer_k_reduce_scatter_launch`
- `dsv4_cp_sparse_attention_backward`
- `dsv4_cp_attention_kv_reduce_scatter_launch`
- `dsv4_cp_local_indexer_grads`
- `dsv4_cp_indexer_k_reduce_scatter_consumer_wait`
- `dsv4_cp_attention_kv_reduce_scatter_consumer_wait`

These Python ranges are visible during eager execution and CUDA Graph capture.
Graph replay does not reissue the host-side NVTX push/pop calls. For a replay
trace, use the outer `DSV4_CSA_*_FORWARD_BACKWARD_*` range together with the
CUDA Graph kernel and collective ordering to verify overlap.

Actual GPU concurrency depends on communication size and host dispatch latency.
An NSYS timeline is the source of truth for whether compute and NCCL overlap in
a particular configuration.
