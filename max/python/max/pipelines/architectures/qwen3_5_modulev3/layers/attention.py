# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/qwen3_5/layers/attention.py
# at main 4df8111afba.
"""Qwen3.5 full attention layer.

Differences from Qwen3 attention:

- ``q_proj`` outputs twice the width, laid out per head as
  ``[head0 Q | head0 gate | head1 Q | head1 gate | ...]``. Reading it as
  ``[all Q | all gate]`` compiles, runs, and silently corrupts.
- Only ``partial_rotary_factor`` of each head's dims get RoPE.
- The Q/K RMSNorms scale by ``1 + weight``.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn._graph_ops import (
    rope_ragged_with_position_ids as _rope_ragged_with_position_ids,
)
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.kernels import (
    flash_attention_ragged,
    flash_attention_ragged_gpu,
    fused_qk_ragged_rope,
    rope_ragged,
    rope_split_store_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Partial

from .linear import (
    Projection,
    ProjectionTensors,
    matmul,
    stacked_matmul,
    weight_rows,
)
from .norm import RMSNorm, rms_norm_scaled

__all__ = ["Qwen3_5Attention"]

# ``max.nn.kernels`` wraps ``rope_ragged`` and the fused rope-and-store ops
# and exports no ``Tensor`` wrapper for
# ``rope_ragged_with_position_ids``. ``F.functional`` supplies that wrapper,
# so each value in ``_attention_core`` stays a ``Tensor``.
rope_ragged_with_position_ids = F.functional(_rope_ragged_with_position_ids)


def _full_width_freqs(
    freqs_cis: Tensor, dtype: DType, head_dim: int, rotary_dim: int
) -> Tensor:
    """Widens the partial-rotary table to ``head_dim`` for the fused store.

    The fused rope-and-store rotates every lane, while Qwen3.5 rotates only
    the last ``rotary_dim``. Prepending the identity rotation ``(cos, sin) =
    (1, 0)`` over the other lanes reproduces partial rotary exactly.
    """
    freqs_cis = freqs_cis.cast(dtype)
    nope_dim = head_dim - rotary_dim
    if nope_dim == 0:
        return freqs_cis
    identity = F.constant(
        np.tile([1.0, 0.0], nope_dim // 2).astype(np.float32),
        DType.float32,
        device=freqs_cis.device,
    ).cast(dtype)
    identity = F.broadcast_to(
        identity.unsqueeze(0), [freqs_cis.shape[0], nope_dim]
    )
    return F.concat([identity, freqs_cis], axis=-1)


def _rope_lanes_last(x: Tensor, n_heads: int, rotary_dim: int) -> Tensor:
    """Moves each head's rotary lanes last and interleaves them.

    The checkpoint puts the rotary lanes first, as halves, while the kernel
    rotates the last lanes as adjacent pairs. ``[x0..x31, x32..x63]`` becomes
    ``[x0, x32, x1, x33, ...]`` after the other lanes.
    """
    rope = x[:, :, :rotary_dim]
    rest = x[:, :, rotary_dim:]
    interleaved = (
        rope.reshape([-1, n_heads, 2, rotary_dim // 2])
        .transpose(-1, -2)
        .reshape([-1, n_heads, rotary_dim])
    )
    return F.concat([rest, interleaved], axis=-1)


def _require_whole_row_labels(kv_params: KVCacheParams) -> None:
    """Refuses labels on a cache that can hold a prefix this step did not write.

    The labeled branch reads no KV cache, so it is right only for a row the
    step wrote whole. A prefix-cache hit leaves ``cache_lengths`` above 0 for
    that row, and the kernel attends over the rows it was handed: the keys an
    earlier step stored are absent from the comparison.

    Args:
        kv_params: This layer's cache parameters.

    Raises:
        ValueError: If ``enable_prefix_caching`` is set.
    """
    if not kv_params.enable_prefix_caching:
        return
    raise ValueError(
        "segment_ids run through flash_attention_ragged_gpu, which reads no"
        " KV cache, so the layer can only attend a row this step wrote whole;"
        " enable_prefix_caching lets a step resume pages an earlier step"
        " wrote, and the kernel sees no key for those tokens. Serve labels"
        " with prefix caching off, or teach this branch to read the cached"
        " keys."
    )


def _attention_core(
    x: Tensor,
    qkv: tuple[ProjectionTensors, ProjectionTensors, ProjectionTensors],
    q_norm: Tensor,
    k_norm: Tensor,
    o_proj: ProjectionTensors,
    kv_collection: PagedCacheValues,
    layer_idx: Tensor,
    freqs_cis: Tensor,
    input_row_offsets: Tensor,
    freq_row_ids: Tensor | None,
    *,
    kv_params: KVCacheParams,
    head_dim: int,
    rotary_dim: int,
    scale: float,
    norm_eps: float,
    segment_ids: Tensor | None,
    mask_variant: MHAMaskVariant,
) -> Tensor:
    """Runs one device's heads and returns its ``o_proj`` partial sums.

    ``segment_ids`` holds each token's int32 label over the step. Labels come
    from :meth:`Qwen3_5Attention.forward`, which follows
    ``max/pipelines/architectures/lfm3/layers/attention.py:116-176``: with
    them this runs the cache-less :func:`flash_attention_ragged_gpu`, the
    ragged kernel that takes them, and without them it runs the paged
    :func:`flash_attention_ragged`. ``mask_variant`` is the mask the chosen
    kernel gets.
    """
    total_seq_len = x.shape[0]
    q_dim = weight_rows(qkv[0].weight)
    kv_dim = weight_rows(qkv[1].weight)
    n_heads = q_dim // (2 * head_dim)
    n_kv_heads = kv_dim // head_dim
    norm = functools.partial(rms_norm_scaled, eps=norm_eps)

    q_out, key, value = F.split(
        stacked_matmul(x, qkv), [q_dim, kv_dim, kv_dim], axis=-1
    )
    q_out = q_out.reshape([-1, n_heads, head_dim * 2])
    query = q_out[:, :, :head_dim]
    gate = q_out[:, :, head_dim : head_dim * 2].reshape(
        [-1, n_heads * head_dim]
    )
    # The norm weights align with the checkpoint's lanes, rotary lanes first.
    query = norm(query, q_norm)
    key = norm(key.reshape([-1, n_kv_heads, head_dim]), k_norm)
    value = value.reshape([-1, n_kv_heads, head_dim])
    query = _rope_lanes_last(query, n_heads, rotary_dim)
    key = _rope_lanes_last(key, n_kv_heads, rotary_dim)

    if kv_params.is_fp8_kv_dtype:
        # The fused rope and store ropes in registers and casts once, where
        # roping K in an FP8 cache would round twice. Q comes out in the
        # cache dtype, which the FP8 attention kernel takes.
        flat = F.concat(
            [
                query.reshape([total_seq_len, -1]),
                key.reshape([total_seq_len, -1]),
                value.reshape([total_seq_len, -1]),
            ],
            axis=-1,
        )
        query = rope_split_store_ragged(
            kv_params=kv_params,
            qkv=flat,
            input_row_offsets=input_row_offsets,
            freqs_cis=_full_width_freqs(
                freqs_cis, flat.dtype, head_dim, rotary_dim
            ),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=n_heads,
            interleaved=True,
            position_ids=freq_row_ids,
            q_out_dtype=kv_params.dtype,
        ).reshape([total_seq_len, n_heads, head_dim])
    else:
        store_k_cache_ragged(kv_collection, key, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, value, input_row_offsets, layer_idx)
        query = fused_qk_ragged_rope(
            kv_params,
            query,
            input_row_offsets,
            kv_collection,
            freqs_cis.cast(query.dtype),
            layer_idx,
            interleaved=True,
            position_ids=freq_row_ids,
        )

    if segment_ids is not None:
        # The paged kernel takes no labels, so this branch reads no cache:
        # ``flash_attention_ragged_gpu`` takes Q, K and V as tensors, and
        # ``segment_ids`` replaces the mask outright. K gets roped here, in
        # registers, over the same lane order the fused store used and the same
        # positions the cache rope used: ``freq_row_ids`` when the step brings
        # it, the row's ``cache_lengths`` entry otherwise.
        _require_whole_row_labels(kv_params)
        if mask_variant is not MHAMaskVariant.NULL_MASK:
            raise ValueError(
                "segment_ids replaces the mask, and"
                " flash_attention_ragged_gpu takes labels only under"
                f" NULL_MASK; this layer asked for {mask_variant} with labels,"
                " a mask the chosen kernel cannot apply"
            )
        if query.dtype != x.dtype:
            raise ValueError(
                "the two attention branches disagree about the output dtype:"
                f" the paged one answers in x.dtype ({x.dtype}) and the"
                f" label one answers in the query's dtype ({query.dtype},"
                f" which follows the {kv_params.dtype} cache it was read"
                " from). Serve labels with the cache dtype and the model"
                " dtype equal, or cast this branch's output."
            )
        if freq_row_ids is not None:
            # One uint32 per token, the operand the two cache ropes above
            # already read, so a step whose positions restart mid-row ropes
            # each token at its own row of ``freqs_cis``.
            key_rows = rope_ragged_with_position_ids(
                key,
                freqs_cis.cast(query.dtype),
                freq_row_ids,
                interleaved=True,
            ).rebind_mapping(key.mapping)
        else:
            key_rows = rope_ragged(
                key,
                input_row_offsets,
                kv_collection.cache_lengths,
                freqs_cis.cast(query.dtype),
                interleaved=True,
            ).rebind_mapping(key.mapping)
        attn_out = flash_attention_ragged_gpu(
            query,
            key_rows,
            value,
            input_row_offsets=input_row_offsets,
            max_seq_len=kv_collection.max_prompt_length,
            mask_variant=mask_variant,
            scale=scale,
            segment_ids=segment_ids,
        )
    else:
        attn_out = flash_attention_ragged(
            kv_params,
            input=query,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=mask_variant,
            scale=scale,
            output_dtype=x.dtype,
        )
    attn_out = attn_out.reshape([total_seq_len, -1])
    return matmul(attn_out * F.sigmoid(gate), o_proj)


class Qwen3_5Attention(Module[..., Tensor]):
    """Gated full attention with partial RoPE, split by head across devices.

    Returns ``o_proj``'s partial sums, which the block reduces.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_dim: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType,
        norm_dtype: DType,
        norm_eps: float,
        scale: float | None = None,
        quant_config: QuantConfig | None = None,
    ) -> None:
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.norm_eps = norm_eps
        self.scale = scale if scale is not None else math.sqrt(1.0 / head_dim)
        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        # Each head's `[query | gate]` block is contiguous, so splitting the
        # rows evenly keeps every gate with its query.
        def head_rows(out_dim: int) -> Projection:
            return Projection(
                hidden_size,
                out_dim,
                dtype=dtype,
                quant_config=quant_config,
                sharding=(TP, None),
            )

        self.q_proj = head_rows(q_dim * 2)
        self.k_proj = head_rows(kv_dim)
        self.v_proj = head_rows(kv_dim)
        self.o_proj = Projection(
            q_dim,
            hidden_size,
            dtype=dtype,
            quant_config=quant_config,
            sharding=(None, TP),
        )
        self.q_norm = RMSNorm(head_dim, eps=norm_eps, dtype=norm_dtype)
        self.k_norm = RMSNorm(head_dim, eps=norm_eps, dtype=norm_dtype)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        freq_row_ids: Tensor | None = None,
        segment_ids: Tensor | None = None,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
    ) -> Tensor:
        """Attends over the row, or over the step's labeled segments.

        Args:
            x: The normed hidden states, ``[total_seq_len, hidden_size]``.
            kv_collection: The attention layers' cache.
            layer_idx: This layer's index in the cache, a host scalar.
            freqs_cis: The rotary table, by position or, with
                ``freq_row_ids``, by token.
            input_row_offsets: The ragged row offsets.
            freq_row_ids: ``[1, total_seq_len]`` row of ``freqs_cis`` per
                token, when the table is per token (M-RoPE) or when a caller
                numbers the step's tokens itself. ``None`` leaves the kernels
                on ``cache_length + token_idx``.
            segment_ids: ``[total_seq_len]`` int32 label per token, where -1 is
                visible to every token of the row and any other label is
                visible inside its own segment. ``None`` runs the paged
                :func:`flash_attention_ragged`; labels run the cache-less
                :func:`flash_attention_ragged_gpu`, which reads no cache.
            mask_variant: The mask the chosen kernel gets.

        Labels come with two preconditions, and this build enforces the first.
        A labeled step has to be one whole row: ``enable_prefix_caching`` lets
        a step resume a prefix another step wrote, and the cache-less kernel
        sees no key for tokens an earlier step stored. The second belongs to
        the caller: a step that traces
        labels inside a layer subgraph cannot see the label tensor built in the
        enclosing graph, so a caller that builds labels outside the layer sets
        ``use_subgraphs = False``, the way a speculative verify and a
        folded-weight build do.

        Raises:
            ValueError: When labels arrive with prefix caching on, with a mask
                the cache-less kernel cannot pair with them, or from a cache
                dtype that leaves the two branches' outputs unequal.
        """
        core = functools.partial(
            _attention_core,
            kv_params=self.kv_params,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            scale=self.scale,
            norm_eps=self.norm_eps,
            segment_ids=segment_ids,
            mask_variant=mask_variant,
        )
        return F.call_on_mesh(
            core,
            x.mesh,
            out_specs=DeviceMapping(x.mesh, (Partial(),)),
        )(
            x,
            (self.q_proj.tensors, self.k_proj.tensors, self.v_proj.tensors),
            self.q_norm.weight,
            self.k_norm.weight,
            self.o_proj.tensors,
            kv_collection,
            layer_idx,
            freqs_cis,
            input_row_offsets,
            freq_row_ids,
        )
