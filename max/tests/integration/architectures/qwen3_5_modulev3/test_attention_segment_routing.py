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
"""Which attention kernel the Qwen3.5 ModuleV3 layer names, read off the graph.

The labeled branch takes its keys from registers and reads no KV cache, so it
is right only for a row the step wrote whole. A numeric check on one prefill
cannot tell the two branches apart: into an empty cache both write the same
bytes and both answer the same. What separates them is which kernel the layer
names, with which mask, and the built graph spells both out. ``_step_mlir``
prints every graph it builds, so an assertion can be checked against the text
it read.

The layer is staged on a one-device CPU mesh, which every host has, and no
case compiles or runs a graph. The graph's device attributes therefore read
``cpu`` where a serving graph would say ``cuda``; neither attention branch
reads a device query, so the kernel choice is the one an accelerator gets.
"""

from __future__ import annotations

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.pipelines.architectures.qwen3_5_modulev3.layers.attention import (
    Qwen3_5Attention,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMapping, DeviceMesh

# The geometry is tiny so a case can stage it, and ``rotary_dim`` stays under
# ``head_dim``, the partial-rotary split whose lane order the layer assumes.
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16
ROTARY_DIM = 8
FREQ_ROWS = 32

DEVICE = DeviceRef.CPU()
MESH = DeviceMesh((CPU(),), (1,), ("tp",))
TOTAL_SEQ_LEN = "total_seq_len"
ROW_OFFSETS_LEN = "batch_plus_one"

# The op names the two attention branches can reach, as the built graph spells
# them. ``flash_attention_ragged`` names the paged kernel and
# ``flash_attention_ragged_gpu`` the cache-less one;
# ``rope_ragged_with_position_ids`` and ``rope_ragged`` are the two ways the
# labeled branch ropes its keys.
PAGED_ATTENTION = 'symbol = "attention.mha.ragged.paged"'
CACHE_LESS_ATTENTION = 'symbol = "attention.mha.ragged.no_cache"'
PER_TOKEN_KEY_ROPE = 'symbol = "rope.ragged.with_position_id"'
RANK_ONE_KEY_ROPE = "mo.composite.rope.ragged"


class AttentionStep(Module[..., Tensor]):
    """An attention layer's step, with its cache rebuilt from graph inputs.

    No graph input is a :class:`PagedCacheValues`: the decoder declares the
    cache as flattened per-device leaves, so a caller of
    :meth:`Qwen3_5Attention.forward` rebuilds the collection itself the way
    ``Qwen3_5TextModel.forward`` does.

    ``labels`` and ``mask_variant`` decide which arguments the layer is called
    with. An unlabeled step is called with the six arguments a build that has
    no ``segment_ids`` parameter accepts, so those cases trace against such a
    build too.
    """

    def __init__(
        self,
        attention: Qwen3_5Attention,
        kv_params: KVCacheParams,
        mesh: DeviceMesh,
        labels: bool,
        mask_variant: MHAMaskVariant | None,
        position_rows: bool,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.kv_params = kv_params
        self.mesh = mesh
        self.labels = labels
        self.mask_variant = mask_variant
        self.position_rows = position_rows

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        freq_row_ids: Tensor,
        segment_ids: Tensor,
        *kv: Tensor,
    ) -> Tensor:
        """Runs the layer over one step of ``total_seq_len`` tokens."""
        cache = PagedCacheValues.from_upstream(
            self.kv_params.unflatten_kv_inputs(
                iter(value._graph_value for value in kv)
            ),
            DeviceMapping.replicated(self.mesh),
        )
        # ``Qwen3_5TextModel`` passes the cache layer index: only the
        # full-attention layers get one, so model layer 3 is cache layer 0 and
        # a one-layer stack puts its attention layer at 0.
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        step: dict[str, object] = {}
        if self.mask_variant is not None:
            step["mask_variant"] = self.mask_variant
        if self.labels:
            step["segment_ids"] = segment_ids
        return self.attention(
            x,
            cache,
            layer_idx,
            freqs_cis,
            input_row_offsets,
            freq_row_ids.unsqueeze(0) if self.position_rows else None,
            **step,
        )


def _cache_params(enable_prefix_caching: bool) -> MHAKVCacheParams:
    """The one-layer cache the layer is built with.

    The cache dtype equals the model dtype. The labeled branch raises when they
    differ, since its answer comes in the query's dtype and the paged branch's
    answer in the model dtype.
    """
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HEAD_DIM,
        n_kv_heads=NUM_KV_HEADS,
        num_layers=1,
        devices=[DEVICE],
        enable_prefix_caching=enable_prefix_caching,
    )


def _step_mlir(
    *,
    labels: bool = False,
    mask_variant: MHAMaskVariant | None = None,
    position_rows: bool = True,
    enable_prefix_caching: bool = False,
) -> str:
    """Stages an attention step and returns its MLIR text.

    The layer's refusals arrive while the graph is traced, so
    :class:`ValueError` reaches the caller before any graph exists.

    Args:
        labels: Whether the step is called with per-token ``segment_ids``.
        mask_variant: The mask to pass, or ``None`` to pass no
            ``mask_variant`` argument and take the layer's default.
        position_rows: Whether the step brings the ``[1, total_seq_len]`` row
            of ``freqs_cis`` per token that M-RoPE brings.
        enable_prefix_caching: Whether the cache may resume a prefix an
            earlier step wrote.
    """
    kv_params = _cache_params(enable_prefix_caching)
    with F.lazy(), default_device(MESH), default_dtype(DType.bfloat16):
        attention = Qwen3_5Attention(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            kv_params=kv_params,
            layer_idx=0,
            dtype=DType.bfloat16,
            norm_dtype=DType.float32,
            norm_eps=1e-6,
        )
        step = AttentionStep(
            attention, kv_params, MESH, labels, mask_variant, position_rows
        )
        step.to(MESH)
    staged, _ = step.stage(
        TensorType(DType.bfloat16, [TOTAL_SEQ_LEN, HIDDEN_SIZE], device=DEVICE),
        TensorType(DType.uint32, [ROW_OFFSETS_LEN], device=DEVICE),
        TensorType(
            DType.float32, [FREQ_ROWS, ROTARY_DIM], device=DeviceRef.CPU()
        ),
        TensorType(DType.uint32, [TOTAL_SEQ_LEN], device=DEVICE),
        TensorType(DType.int32, [TOTAL_SEQ_LEN], device=DEVICE),
        *kv_params.flattened_kv_inputs(),
    )
    mlir = str(staged)
    print(
        f"===== staged graph: labels={labels}"
        f" mask_variant={mask_variant} position_rows={position_rows}"
        f" prefix_caching={enable_prefix_caching} ====="
    )
    print(mlir)
    return mlir


def _op_lines(mlir: str, symbol: str) -> list[str]:
    """The graph lines that name ``symbol``."""
    return [line for line in mlir.splitlines() if symbol in line]


def test_labels_name_the_cache_less_op_and_the_segment_mask() -> None:
    """Per-token labels move the step to the kernel that takes them.

    ``flash_attention_ragged`` has no labels parameter, so a step that brings
    labels cannot reach it, and the cache-less op takes the segment mask where
    ``mask_variant`` would go. With per-token rows of ``freqs_cis`` in the
    step, the keys rope through the per-token op; the rank-1 one reads the
    cache's row lengths instead.
    """
    mlir = _step_mlir(labels=True, mask_variant=MHAMaskVariant.NULL_MASK)
    attention_lines = _op_lines(mlir, CACHE_LESS_ATTENTION)
    assert attention_lines, "no cache-less attention op in the graph"
    assert 'mask_str = "segment"' in attention_lines[0], (
        "the cache-less op lost the segment mask: the labels would then be"
        " operands of a kernel that ignores them"
    )
    assert PER_TOKEN_KEY_ROPE in mlir, (
        "the keys are not roped by the per-token rows the step brought"
    )
    assert PAGED_ATTENTION not in mlir, (
        "a labeled step also names the paged kernel, which reads the cache"
        " this step left alone"
    )
    assert RANK_ONE_KEY_ROPE not in mlir, (
        "the keys rope by cache row length, so a step whose positions restart"
        " mid-row ropes them at the wrong row of freqs_cis"
    )


def test_an_unlabeled_step_names_the_paged_op() -> None:
    """Without labels the step keeps the kernel that reads the cache.

    The call names no labeled argument, so a build with no ``segment_ids``
    parameter accepts it and stages the same graph.
    """
    mlir = _step_mlir()
    assert PAGED_ATTENTION in mlir
    assert CACHE_LESS_ATTENTION not in mlir
    assert PER_TOKEN_KEY_ROPE not in mlir


def test_the_layer_refuses_labels_under_prefix_caching() -> None:
    """The cache-less kernel sees no key an earlier step stored.

    A prefix-cache hit leaves ``cache_lengths`` above 0 for the row, and the
    labeled branch reads no cache at all.
    """
    with pytest.raises(ValueError) as excinfo:
        _step_mlir(
            labels=True,
            mask_variant=MHAMaskVariant.NULL_MASK,
            enable_prefix_caching=True,
        )
    assert "enable_prefix_caching" in str(excinfo.value)


def test_the_layer_refuses_labels_with_a_causal_mask() -> None:
    """``segment_ids`` replaces the mask, so the two cannot arrive together.

    The message names ``NULL_MASK``, the only mask the cache-less kernel pairs
    with labels, and the mask the layer asked for.
    """
    with pytest.raises(ValueError) as excinfo:
        _step_mlir(labels=True, mask_variant=MHAMaskVariant.CAUSAL_MASK)
    message = str(excinfo.value)
    assert "NULL_MASK" in message
    assert "CAUSAL_MASK" in message


def test_the_default_mask_variant_keeps_the_causal_mask() -> None:
    """An unlabeled step that names no mask still attends causally.

    ``mask_variant`` defaults to ``MHAMaskVariant.CAUSAL_MASK``, so the paged
    op keeps the mask it had before the parameter existed.
    """
    mlir = _step_mlir()
    attention_lines = _op_lines(mlir, PAGED_ATTENTION)
    assert attention_lines, "no paged attention op in the graph"
    assert 'mask_str = "causal"' in attention_lines[0]
    assert 'mask_str = "null"' not in mlir


def test_labels_without_position_rows_rope_keys_from_the_cache_row() -> None:
    """A step with no rows of ``freqs_cis`` per token ropes by row length.

    The labeled branch has two ways to rope its keys, and the rank-1 one is
    what a plain prefill takes. Pinning it keeps the ``RANK_ONE_KEY_ROPE``
    absence in
    :func:`test_labels_name_the_cache_less_op_and_the_segment_mask` from
    matching a name that reaches no graph.
    """
    mlir = _step_mlir(
        labels=True,
        mask_variant=MHAMaskVariant.NULL_MASK,
        position_rows=False,
    )
    assert RANK_ONE_KEY_ROPE in mlir
    assert PER_TOKEN_KEY_ROPE not in mlir
