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

"""The gated DeltaNet with its five projection deltas applied.

What this file is: the mixer the Locate fold installs in the 24
``linear_attention`` blocks of the decider's 32-layer hybrid tower
(:class:`LocateGatedDeltaNet`). Its 120 ``(lora_a, lora_b)`` pairs are 44,924,928
of the adapter's 129,859,584 float32 elements, and the wrapped-projection seam of
:mod:`.adapter` reaches none of them.

Derived from
``max.pipelines.architectures.qwen3_5_modulev3.layers.gated_deltanet``
(:class:`GatedDeltaNet`, ``_gated_delta_core``). The projection declarations, the
on-host ``in_proj_qkv`` placed per device by ``shard_segments``, the decay and beta
terms, the slot-indexed conv and recurrence kernels, the SM100 gate that picks the
chunked prefill kernel over the sequential one, the ``F.cond`` split between the
two, the gated norm that scales by ``weight`` and reduces inside one head, the
swish output gate, and the ``call_on_mesh`` / ``Partial()`` shape of ``forward``
are the donor's, reached through the donor's class and helpers.

Deltas applied, and nothing else:

- **Each of the five projections gets its adapter delta** at the point where the
  core splits or ends its matmul: ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``
  and ``in_proj_a`` after ``stacked_matmul`` and ``F.split`` (donor ``:213-224``),
  ``out_proj`` after the gated matmul (donor ``:451``). Each delta is
  ``x_f32 @ A.T @ B.T`` in float32, added once and cast back, as
  ``LoRALinear.forward`` arranges it (``locate.py:83-92``).
- **The speculative verify path is dropped.** A verify passes ``verify_width`` and
  a ``ReplaySink``, runs the ring recurrence and the ``chunk_long_rows`` map, and
  records replay inputs for the rollback (donor ``:245-264``, ``:266-276``,
  ``:322-373``, ``:398-436``). This port registers no speculative target, so
  :meth:`LocateGatedDeltaNet.forward` raises when ``self.verify`` is set instead
  of carrying that half of the body.

Why the per-device body is re-declared and not subclassed around: the donor's body
is the module-level function ``_gated_delta_core`` (donor ``:182-451``), which its
``forward`` (donor ``:565``) binds by name out of the donor's own module globals at
``:584-593`` and calls inside the single ``F.call_on_mesh(...)(...)`` expression at
``:594-611``. No class attribute names that function, so a subclass cannot rebind
it, and there is no separate core call a subclass could wrap. The projections
themselves are modules (donor ``:519-538``, ``:555-561``) but ``forward`` reads
only their ``.tensors`` at ``:574-582``, and ``stacked_matmul`` concatenates the
four input-side weights and splits the one matmul inside the body, so a subclass
of ``Projection`` overriding ``forward`` changes nothing. The only seam at that
boundary is the tensors themselves: handing the core a ``ProjectionTensors`` whose
weight is ``base + B @ A`` in float32 costs one dense float32 ``[out, in]`` per
stacked group per layer: 42,106,880 elements per layer, 1,010,565,120 over the
24 layers, 4,042 MB on top of the 8.46 GiB the checkpoint's safetensors headers
report. ``stacked_matmul`` concatenates weights, so that merge also moves the
DeltaNet projections' 1.01 G MACs per token from bfloat16 into float32. That price
buys the same 120 sites this copy runs for 44.9 M added MACs per token and no
weight bytes, so the body is copied. Upstream fix worth filing: a ``deltas``
argument on ``_gated_delta_core`` and ``GatedDeltaNet``, after which this file is
one subclass with no body.

Where the deltas sit against the recurrence scale: MAX's recurrence output equals
the reference's times ``1 / sqrt(128)`` = 0.0883882, and ``rms_norm_scaled``
(donor ``:443``) cancels that factor because it normalizes per value head. The four
input-side deltas are inside the quantity the kernels scale, so they get the same
factor as the base and the norm cancels both; a delta added to ``recurrence``
before that norm would be off by 0.0883882. The ``out_proj`` delta is added to
``gated``, which is post-norm, where the two routes already agree.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from max import functional as F
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorType, TensorValue
from max.nn._graph_ops.state_space import read_state_rows, write_state_rows
from max.nn.kernels import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
    kda_chunk,
    kda_chunk_is_registered,
    kda_chunk_supports_head_dims,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.qwen3_5_modulev3.layers.gated_deltanet import (
    _CHUNK_MIN_COMPUTE_CAPABILITY,
    _OUTPUT_GATE_ACTIVATIONS,
    GatedDeltaNet,
    GatedDeltaStateAccess,
    _nvidia_compute_capability,
)
from max.pipelines.architectures.qwen3_5_modulev3.layers.linear import (
    ProjectionTensors,
    matmul,
    stacked_matmul,
)
from max.pipelines.architectures.qwen3_5_modulev3.layers.norm import (
    rms_norm_scaled,
)
from max.pipelines.architectures.qwen3_5_modulev3.layers.sharding import (
    shard_segments,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Partial

from .adapter import delta, pair_of, with_delta

__all__ = [
    "RESTART_BLOCKS",
    "STATE_RESTART_ENABLED",
    "DeltaSites",
    "LocateGatedDeltaNet",
    "StateRestart",
    "attention_labels",
    "restart_edges",
    "restart_segment_index",
    "segment_index_from_edges",
]

logger = logging.getLogger(__name__)

STATE_RESTART_ENABLED = os.environ.get(
    "DECIDER_STATE_RESTART", "1"
).lower() in ("1", "on", "true")
"""The gate for rewinding the recurrent pools between question blocks: on
unless the environment names ``0``, ``off`` or ``false``.

On, a row that carries the state once and one question chunk per question
rewinds the recurrent pool to the end of the state before each chunk and leaves
the conv window at the end of the state for the whole step, so question two
reads no part of question one. Off, each layer runs its step's tokens in one
conv and one recurrence call, the route ``mask_off.log`` and ``sq_cost.log``
measured. Read from the environment once per process at import, because the
segment split is part of the compiled graph. ``LAYOUT_DEFAULT_NOTES.md`` records
why the build starts with it on.
"""

RESTART_BLOCKS = int(os.environ.get("DECIDER_RESTART_BLOCKS", "3"))
"""How many question blocks one compiled graph rewinds between.

The graph traces one conv and one recurrence call per block plus one for the
state, so the count is baked in at build, and the edge table it declares is
``[blocks + 2]`` long: ``blocks + 1`` spans. A request that states more spans
than that meets :meth:`~.batch_processor.SegmentBounds.edges`, which refuses it
by name. The counted fallback, :func:`restart_edges`, folds an unexpected extra
chunk into the last span. The recorded cases top out at three questions
(``multi-question``), which is the default, and the last span's end edge is the
step's token count, so every token lands inside a span.
"""


DeltaSites = tuple[
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    Tensor | None,
]
"""The six delta slots :func:`_gated_delta_core_with_deltas` reads, in order.

``(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, gate)``: each pair is
that projection's ``(lora_a, lora_b)``, and ``gate`` is the step's per-token
adapter gate. :meth:`LocateGatedDeltaNet.forward` hands ``None`` when no adapter
is bound, and the core then leaves every projection at its base term.
"""


@dataclass
class StateRestart:
    """The step's segment table, shared by one graph's linear layers.

    One instance per compiled graph, bound once per step by
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderDecoder.forward`, the way
    :class:`.adapter.AdapterGate` carries the adapter gate.
    """

    value: Tensor | None = None
    """``[blocks + 2]`` int32 token edges, or ``None`` before the step bound
    them.

    The step supplies this tensor as a graph input, and the renderer's host
    rule for it is :func:`restart_edges`. A build never reduces the step's
    segment labels into these edges: a value computed inside the graph from a
    ``Dim('total_seq_len')`` reduction and handed to ``gated_delta_conv1d_fwd``
    or ``gated_delta_recurrence_fwd`` as ``input_row_offsets`` faults in
    ``_Model_execute`` (``RESTART_DEBUG_NOTES.md``, reproduced by
    ``spike/restart_debug/table_probe.py --mode table --named``).
    """

    segment_index: Tensor | None = None
    """``[total_seq_len]`` int32 segment index of each token, the operand the
    per-segment outputs are assembled with."""

    positions: Tensor | None = None
    """``[1, total_seq_len]`` uint32 rope position of each token, or ``None``.

    Bound under :data:`~.attention.POSITION_RESTART_ENABLED` only: the eight
    full-attention layers rope at these values, which restart each chunk at the
    end of the state. ``None`` leaves MAX numbering tokens from the row
    boundaries.
    """

    blocks: int = RESTART_BLOCKS
    """Question blocks this graph traces, from :data:`RESTART_BLOCKS`."""


def restart_edges(labels: Sequence[int], blocks: int) -> list[int]:
    """Returns the ``[blocks + 2]`` token edges of one row, counted on the host.

    ``labels`` is the row's segment label per token, the renderer's
    ``StateOncePlan.segment_ids``: -1 for the state block and the separator that
    closes it, ``i`` for question chunk ``i``. The edge before segment ``j``
    counts the tokens whose label is below ``j``, so segment 0 is the state,
    segment ``i + 1`` is chunk ``i``, and the last edge is the row's token count,
    which folds an unexpected extra chunk into the last span.

    A row with fewer chunks than ``blocks`` repeats its token count in the extra
    edges, and those spans are empty: the kernels walk them and read no rows.

    The renderer's labels restart at every row, so the table describes one row
    per step, which is what the decisions route scores.

    The fallback rule: a request that states its spans reaches the step as data
    (:class:`~qwen3_5_decider.batch_processor.SegmentBounds`), and this count
    runs only when a context states none.
    """
    total = len(labels)
    edges: list[int] = [0]
    seen = 0
    for label in range(-1, blocks - 1):
        seen += sum(1 for token_label in labels if token_label == label)
        edges.append(seen)
    edges.append(total)
    return edges


def segment_index_from_edges(
    edges: Tensor, total_tokens: Dim | int, blocks: int
) -> Tensor:
    """Returns the per-token segment index of the step, read off the edges.

    ``edges`` is the step's ``[blocks + 2]`` token edge table
    (:func:`restart_edges`), which arrives as a graph input and says where the
    state block ends and where each question chunk ends. A token belongs to the
    last span that starts at or before it, so its segment index is the number of
    edges at or below its token index, less one, held inside ``[0, blocks]``:
    segment 0 is the state block and segment ``i + 1`` is question chunk ``i``.
    An empty span repeats an edge, and the later span wins that token, which is
    how the kernels walk a span of no tokens.

    This is the boundary-as-data path: the spans come from the request's own
    encoders (``StateOncePlan.segment_ids``, ``LocatePlan.segments``) through
    :meth:`qwen3_5_decider.batch_processor.Qwen3_5DeciderBatchProcessor.token_positions`,
    so a sequence whose chunk ends are not ``<|fim_suffix|>`` still gets one span
    per question. :func:`restart_segment_index` is the fallback, which counts
    that token per chunk.
    """
    token_index = F.range(
        0,
        total_tokens,
        1,
        out_dim=total_tokens,
        dtype=DType.int32,
        device=edges.device,
    )
    # One comparison per (token, edge) pair: the count of edges a token has
    # passed is its segment index. The reduction runs along the static edge axis,
    # and its result feeds the mask and the per-segment assembly; the state
    # kernels get their offsets from the graph input alone, which is where
    # ``RESTART_DEBUG_NOTES.md`` says they have to come from.
    starts = F.reshape(edges, [1, edges.shape[0]])
    passed = F.cast(
        F.greater_equal(F.reshape(token_index, [-1, 1]), starts), DType.int32
    )
    # ``F.sum`` along an axis keeps that axis at size 1, and the attention kernel
    # takes a rank-1 label per token.
    counts = F.reshape(F.sum(passed, axis=1), [-1])
    return F.clamp(counts - 1, 0, blocks)


def attention_labels(segment_index: Tensor) -> Tensor:
    """Returns the step's label per token for the segment-masked attention.

    ``segment_index`` is :func:`segment_index_from_edges`' or
    :func:`restart_segment_index`' index, and the labels are that index less
    one: the state block is the shared ``-1`` the kernel shows to every token of
    the row, and question chunk ``i`` is ``i``. Labels stay nondecreasing inside
    a row, which is what ``flash_attention_ragged_gpu`` asks for.
    """
    return segment_index - 1


def restart_segment_index(labels: Tensor, blocks: int) -> Tensor:
    """Returns the per-token segment index of the step, over ``labels``.

    ``labels`` is :func:`qwen3_5_decider.layers.attention.marker_segments`'
    label per token, and the index it answers is the operand the per-segment
    kernel outputs are assembled with (``restarted_states``). Label -1 is
    segment 0 and chunk ``i`` is segment ``i + 1``, capped at ``blocks`` because
    the last span runs to the end of the step. Without the cap a token in an
    extra chunk is assembled from no span at all and its mixer output is zero:
    at ``blocks = 0`` that is every token past the state block.
    """
    return F.elementwise_min(F.cast(labels, DType.int32) + 1, blocks)


def _snapshot_rows(pool: BufferValue, rows: TensorValue) -> TensorValue:
    """Returns ``pool[rows]``, the read half of MAX's state-row copy."""
    return read_state_rows(pool, rows)


snapshot_state_rows = F.functional(_snapshot_rows)


def _restore_rows(
    pool: BufferValue, rows: TensorValue, values: TensorValue
) -> None:
    """Writes ``values`` into ``rows`` of ``pool``, the write half of the copy."""
    write_state_rows(pool, rows, values)


restore_state_rows = F.functional(_restore_rows)


def _gated_delta_core_with_deltas(
    x: Tensor,
    in_proj: Sequence[ProjectionTensors],
    in_proj_ba: Sequence[ProjectionTensors] | None,
    conv_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm_weight: Tensor,
    out_proj: ProjectionTensors,
    adapter: DeltaSites | None,
    state: GatedDeltaStateAccess,
    input_row_offsets: Tensor,
    restart: StateRestart | None,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    ssm_dtype: DType,
    norm_eps: float,
    output_gate: Callable[[Tensor], Tensor],
) -> Tensor:
    """Runs one device's heads and returns its ``out_proj`` partial sums.

    A per-device body, run by :meth:`LocateGatedDeltaNet.forward` through
    :func:`max.functional.call_on_mesh` as the donor does, so each device owns
    its share of the heads and the sharding solver inserts the all-reduce the
    row-parallel output needs. Structurally the donor's ``_gated_delta_core``;
    the five deltas and the absent verify path are the port's.

    The donor's five projections reach this body as bare tensors, so no wrapped
    projection ``forward`` runs: each delta is added where this function splits
    or ends its matmul, the same place the base term comes from. ``adapter`` is
    ``None`` on a build with no adapter bound, and the six slots of a
    :data:`DeltaSites` on one that has.
    """
    device = x.device
    nv = num_value_heads
    vd = value_head_dim
    key_dim = key_head_dim * num_key_heads
    value_dim = value_head_dim * num_value_heads
    conv_dim = key_dim * 2 + value_dim
    kernel_size = int(conv_weight.shape[-1])

    if in_proj_ba is None:
        qkv, z, b_proj, a_proj = F.split(
            stacked_matmul(x, in_proj), [conv_dim, value_dim, nv, nv], axis=-1
        )
    else:
        qkv, z = F.split(
            stacked_matmul(x, in_proj), [conv_dim, value_dim], axis=-1
        )
        b_proj, a_proj = F.split(
            stacked_matmul(x, in_proj_ba), [nv, nv], axis=-1
        )
    if adapter is not None:
        qkv_pair, z_pair, b_pair, a_pair, out_pair, adapter_gate = adapter
        if adapter_gate is not None:
            # ``F.split`` of the stacked base matmul is row-for-row the four base
            # projections, so adding each delta after the split leaves the
            # gated-off bits alone and matches ``LoRALinear.forward`` for a
            # gated row.
            qkv = qkv + delta(x, qkv_pair, adapter_gate)
            z = z + delta(x, z_pair, adapter_gate)
            b_proj = b_proj + delta(x, b_pair, adapter_gate)
            a_proj = a_proj + delta(x, a_pair, adapter_gate)
    qkv_f32 = qkv.cast(DType.float32)

    dt_bias = dt_bias.cast(DType.float32)
    a_log = a_log.cast(DType.float32)
    decay_rate = F.exp(a_log.cast(ssm_dtype))
    a_float = a_proj.cast(ssm_dtype)
    # A stable softplus: past 20 it is the identity, and exp would overflow.
    x_sp = a_float + dt_bias.cast(ssm_dtype)
    softplus = F.where(x_sp > 20.0, x_sp, F.log(1.0 + F.exp(x_sp)))
    decay = F.exp((-decay_rate * softplus).cast(DType.float32))
    beta = F.sigmoid(b_proj).cast(DType.float32)

    conv_weight = conv_weight.cast(DType.float32).reshape(
        [conv_dim, kernel_size]
    )
    offsets_uint32 = input_row_offsets.cast(DType.uint32)
    offsets_int32 = input_row_offsets.cast(DType.int32)
    conv_slot = state.conv_row_id.cast(DType.uint32)
    rec_slot_uint32 = state.recurrent_row_id.cast(DType.uint32)
    rec_slot_int32 = state.recurrent_row_id.cast(DType.int32)
    if restart is not None:
        # The restarted path re-derives both row ids through ``_bounded_row_id``
        # (see that function for why an out-of-range index needs a check here).
        conv_slot = _bounded_row_id(state.conv_pool, state.conv_row_id)
        rec_slot_uint32 = _bounded_row_id(
            state.recurrent_pool, state.recurrent_row_id
        )
        rec_slot_int32 = rec_slot_uint32.cast(DType.int32)

    if restart is not None:
        recurrence = restarted_states(
            qkv_f32, decay, beta, conv_weight, state, restart, value_dim
        )
    else:
        conv_output = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_f32,
            conv_weight=conv_weight,
            conv_state=state.conv_pool,
            slot_idx=conv_slot,
            input_row_offsets=offsets_uint32,
        )
        conv_output = F.silu(conv_output)

    def chunk_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Separate bf16 q/k/v give the chunk op explicit head dims.
        q, k, v = F.split(conv_output, [key_dim, key_dim, value_dim], axis=-1)
        q = q.reshape([-1, num_key_heads, key_head_dim]).cast(DType.bfloat16)
        k = k.reshape([-1, num_key_heads, key_head_dim]).cast(DType.bfloat16)
        v = v.reshape([-1, nv, vd]).cast(DType.bfloat16)
        # The chunk op takes a per-channel gate, so each head's scalar is
        # spread along the key dim. A multiply materializes it: the kernel
        # reads the elements through a TMA descriptor, which a stride-0
        # broadcast view would not back.
        ones_k = F.constant(
            np.ones((1, 1, key_head_dim), dtype=np.float32),
            DType.float32,
            device=device,
        )
        raw_gate = a_float.reshape([-1, nv, 1]) * ones_k
        dt_bias_2d = dt_bias.reshape([nv, 1]) * F.constant(
            np.ones((1, key_head_dim), dtype=np.float32),
            DType.float32,
            device=device,
        )
        return q, k, v, raw_gate, dt_bias_2d

    def prefill_recurrence() -> Tensor:
        q, k, v, raw_gate, dt_bias_2d = chunk_inputs()
        out = kda_chunk(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            beta_logits=beta,
            a_log=a_log,
            dt_bias=dt_bias_2d,
            cu_seqlens=offsets_int32,
            state_pool=state.recurrent_pool,
            state_indices=rec_slot_int32,
            output_dtype=DType.float32,
            gate_mode="original",
            beta_mode="probability",
            state_layout="K_FIRST",
            use_computebound=True,
        )
        return out.reshape([-1, value_dim])

    def decode_recurrence() -> Tensor:
        return gated_delta_recurrence_fwd(
            qkv_conv_output=conv_output,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=state.recurrent_pool,
            slot_idx=rec_slot_uint32,
            input_row_offsets=offsets_uint32,
        )

    # The chunk op's kernels are written for NVIDIA SM100+ (tcgen05 MMAs and
    # TMA), and a build without the closed-source kernels doesn't register
    # it; everything else runs the sequential recurrence for prefill too.
    # TODO(KERN-3635): drop this gate once kda_chunk compiles below SM100.
    chunk_servable = (
        accelerator_api() == "cuda"
        and _nvidia_compute_capability() >= _CHUNK_MIN_COMPUTE_CAPABILITY
        and kda_chunk_supports_head_dims(key_head_dim, value_head_dim)
        and kda_chunk_is_registered()
    )
    output_types = [
        TensorType(
            DType.float32,
            [x.shape[0], value_dim],
            DeviceRef.from_device(device),
        )
    ]
    total_tokens = F.shape_to_tensor(x.shape)[0]
    batch_size = F.shape_to_tensor(input_row_offsets.shape)[0] - 1
    # A decode batch has one token per sequence. `F.cond` branches are
    # separate regions, so a device graph captured for decode records only
    # the decode kernels, and not the chunk map's host readback.
    is_prefill = total_tokens > batch_size
    if restart is not None:
        # ``restarted_states`` already ran the recurrence, segment by segment.
        # The chunked prefill kernel keeps its own state inside one call over
        # ``cu_seqlens`` and takes no slot index per segment, so the restarted
        # path stays on the sequential recurrence it ran in ``restart_probe``.
        if chunk_servable:
            raise NotImplementedError(
                "the restarted DeltaNet path runs the sequential recurrence;"
                " the chunked SM100 kernel gets no per-segment slot index"
            )
    elif chunk_servable:
        (recurrence,) = F.cond(
            is_prefill,
            output_types,
            lambda: [prefill_recurrence()],
            lambda: [decode_recurrence()],
        )
    else:
        recurrence = decode_recurrence()

    recurrence = F.rebind(
        recurrence,
        [x.shape[0], value_dim],
        "recurrence_output_flat total_seq_len rebind",
    )
    # Normalize, then gate.
    normed = rms_norm_scaled(
        recurrence.reshape([-1, nv, vd]).cast(x.dtype),
        norm_weight,
        eps=norm_eps,
        weight_offset=0.0,
    )
    gate = output_gate(z.reshape([-1, nv, vd]).cast(DType.float32))
    gated = (
        (normed.cast(DType.float32) * gate)
        .cast(x.dtype)
        .reshape([-1, value_dim])
    )
    base_out = matmul(gated, out_proj)
    if adapter is None:
        return base_out
    *_, out_pair, adapter_gate = adapter
    if adapter_gate is None:
        return base_out
    return with_delta(base_out, gated, out_pair, adapter_gate)


def _bounded_row_id(pool: Tensor, row_id: Tensor) -> Tensor:
    """Returns ``row_id`` held inside the row range of ``pool``.

    A state row index is a bare integer and ``state_space_copy_rows`` accepts one
    past the pool: ``restart_probe`` read back whatever sits after the pool and
    wrote there without an error, and ``ISOLATED_NOTES.md`` section 2 ruled out
    row arithmetic for the same reason. Neither the state kernels nor the two
    state-row copies check the index, so every one this layer produces comes
    through here.

    The clamp only runs where the pool's own row count is a host integer: the
    probes in ``spike/restart_debug/`` size their pools that way. A served build
    sizes both pools with a symbolic dim, ``Dim('linear_attn_conv_rows')`` and
    ``Dim('linear_attn_recurrent_rows')``, and no bound can be applied to an
    index against a dim, so the id passes through. The first layer to meet that
    shape logs it once per process; a gated build reaches this line twice per
    linear layer, 48 times over 24 layers.
    """
    rows = pool.shape[0]
    if not isinstance(rows, int):
        global _ROW_BOUND_GAP_LOGGED
        if not _ROW_BOUND_GAP_LOGGED:
            _ROW_BOUND_GAP_LOGGED = True
            logger.warning(
                "the state pool's row count is the dim %r, so no bound can be"
                " applied to a state row id in this build; a row id past the"
                " pool reaches the kernels and the two state-row copies."
                " Reported once per process for the whole build.",
                rows,
            )
        return row_id.cast(DType.uint32)
    return F.clamp(row_id.cast(DType.int32), 0, rows - 1).cast(DType.uint32)


_ROW_BOUND_GAP_LOGGED = False
"""True once a build has logged the unclampable state-row bound."""


def restarted_states(
    qkv_f32: Tensor,
    decay: Tensor,
    beta: Tensor,
    conv_weight: Tensor,
    state: GatedDeltaStateAccess,
    restart: StateRestart,
    value_dim: int,
) -> Tensor:
    """Runs the conv and the recurrence segment by segment on one pool row.

    Segment 0 is the state block, and the rest are the question blocks in row
    order. After segment 0 the recurrent pool is read out with
    ``read_state_rows``; before each later segment that snapshot is written back
    with ``write_state_rows``, so each block starts from the end of the state and
    reads no earlier block's tokens. One row serves the whole step, which is all
    the served path reserves per request (``ISOLATED_NOTES.md`` section 1).

    Only segment 0's conv writes its window, and the conv pool gets no snapshot.
    The conv kernel maps every token of the step to a row of
    ``input_row_offsets``, so a span starting at token ``e`` also runs the tokens
    before ``e``, each at a negative position inside the span, and each of those
    takes the span's head-token branch and stores the span's end window in the
    slot. That store lands before the span's own head token reads the window, so
    the head tokens convolve the previous span's last ``kernel_size - 1`` tokens
    and every later token inherits the wrong state.
    ``spike/restart_debug/span_probe.py`` reads that at ``2.5397e-02`` against
    one call over a row with one question, where the split owes no change, and at
    ``0.000e+00`` once those spans stop writing the window.

    Segment 0 leaves the window holding the end of the state block and no later
    span moves it, which is the look-back every span needs. The conv row at the
    end of the step therefore holds the state block's window: a step resuming
    this row would read look-back tokens from the wrong place, and the
    registration gives every row one step (``RESTART_SERVED_NOTES.md``).

    Each call walks the whole ragged array and writes only the rows its
    ``input_row_offsets`` span, so the per-segment outputs are gathered back
    into one row per token by the segment index of that token.

    The gather is a select, not a masked sum, and that is not a style choice.
    ``gated_delta_recurrence_fwd`` writes the rows of its span and leaves the
    rest of its output buffer as it found it (``max/kernels/src/state_space/
    gated_delta.mojo:403`` walks ``range(sequence_start_flat_idx,
    sequence_end_flat_idx)``), and what an unwritten row holds is whatever the
    allocator last put there. Multiply that buffer by a 0/1 mask and add, and
    every unwritten cell contributes ``NaN * 0``, which is ``NaN``, so a layer
    reports non-finite values for rows it never read. ``spike``-side taps over
    the served 4090 build counted 0 non-finite cells in the rows a call owns
    and 315,400 in the rows a 106-token step left unwritten, at the first
    linear layer, with every answer of that step then refused as non-finite;
    the same build on Apple silicon reads those rows as zero and answers. A
    select takes one branch per element and never touches the other, so an
    unwritten row cannot reach the output of the segment that does not own it.
    """
    table = restart.value
    segment_index = restart.segment_index
    assert table is not None and segment_index is not None, (
        "the restart gate is set and this step bound no segment table"
    )
    offsets_uint32 = table.cast(DType.uint32)
    conv_slot = _bounded_row_id(state.conv_pool, state.conv_row_id)
    rec_slot = _bounded_row_id(state.recurrent_pool, state.recurrent_row_id)

    assembled: Tensor | None = None
    snapshot: Tensor | None = None
    for segment in range(restart.blocks + 1):
        if snapshot is not None:
            restore_state_rows(state.recurrent_pool, rec_slot, snapshot)
        span = offsets_uint32[segment : segment + 2]
        conv_out = F.silu(
            gated_delta_conv1d_fwd(
                qkv_input_ragged=qkv_f32,
                conv_weight=conv_weight,
                conv_state=state.conv_pool,
                slot_idx=conv_slot,
                input_row_offsets=span,
                # The write is segment 0's job; see this function's docstring.
                write_state=segment == 0,
            )
        )
        out = gated_delta_recurrence_fwd(
            qkv_conv_output=conv_out,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=state.recurrent_pool,
            slot_idx=rec_slot,
            input_row_offsets=span,
        )
        out = F.rebind(
            out,
            [qkv_f32.shape[0], value_dim],
            "restarted segment recurrence flat total_seq_len rebind",
        )
        # One row per token, taken from the call whose span covers that token.
        # ``F.where`` for the reason in this function's docstring: the rows this
        # call left unwritten hold allocator garbage, and garbage must never be
        # multiplied into the sum.
        hit = F.reshape(segment_index == segment, [-1, 1])
        assembled = out if assembled is None else F.where(hit, out, assembled)
        if segment == 0 and restart.blocks:
            # The rewind point: the recurrent row as it stands at the end of the
            # state block. A build with no question blocks needs no copy.
            snapshot = snapshot_state_rows(state.recurrent_pool, rec_slot)
    assert assembled is not None, "the loop traces at least one segment"
    return assembled


class LocateGatedDeltaNet(GatedDeltaNet):
    """Qwen3.5 gated DeltaNet that adds each projection's adapter delta.

    The parameters -- ``in_proj_qkv`` kept on the host and segmented per device,
    ``in_proj_z``/``in_proj_b``/``in_proj_a`` split by head, ``out_proj`` split
    by input, and the conv, decay and norm scalars -- are the donor's
    ``__init__`` unchanged, so the layer loads the same checkpoint tensors as the
    mixer it replaces. Only ``forward`` is re-declared, to bind the core that
    reads the five ``(lora_a, lora_b)`` pairs.
    """

    restart: StateRestart | None = None
    """The step's segment table, or ``None`` on a mixer the decoder never wired.

    A wired mixer restarts its state between the question blocks of the step;
    an unwired one runs its step in one call over the row. :meth:`forward`
    passes this to the core, and the core refuses a restarted call whose table
    or segment index is unset.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        rms_norm_eps: float,
        ssm_dtype: DType = DType.float32,
        quant_config: QuantConfig | None = None,
        output_gate_type: str = "swish",
        scalar_dtypes: Mapping[str, DType] | None = None,
    ) -> None:
        """Builds the donor layer, argument for argument.

        The donor's keyword-only signature is spelled out rather than forwarded
        as ``*args`` and ``**kwargs``, so a call site that names an argument the
        mixer does not take fails here with the argument's name.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_size=conv_kernel_size,
            dtype=dtype,
            rms_norm_eps=rms_norm_eps,
            ssm_dtype=ssm_dtype,
            quant_config=quant_config,
            output_gate_type=output_gate_type,
            scalar_dtypes=scalar_dtypes,
        )

    def forward(
        self,
        x: Tensor,
        state: GatedDeltaStateAccess,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs the linear attention over ``x`` and advances ``state``.

        Raises:
            NotImplementedError: If a speculative verify set ``self.verify``.
                The copied body has no ring recurrence and no replay capture.
        """
        if self.verify is not None:
            raise NotImplementedError(
                "LocateGatedDeltaNet has no speculative verify path; this port"
                " registers no speculative target"
            )
        mesh = x.mesh
        num_devices = mesh.num_devices
        qkv = self.in_proj_qkv.tensors
        qkv = ProjectionTensors(
            weight=shard_segments(qkv.weight, self.segments, mesh),
            quant_config=qkv.quant_config,
            weight_scale=qkv.weight_scale,
            input_scale=qkv.input_scale,
        )
        ba = (self.in_proj_b.tensors, self.in_proj_a.tensors)
        in_proj = (qkv, self.in_proj_z.tensors)
        # A decisions build installs no adapter: the checkpoint has no
        # ``*.lora_a`` tensors, so there are no pairs to bind and the core gets
        # ``None`` where a folded build hands it the six slots.
        gate = getattr(self, "gate", None)
        sites: DeltaSites | None = None
        if gate is not None:
            qkv_a, qkv_b = pair_of(self.in_proj_qkv)
            # ``in_proj_qkv`` is the donor's on-host projection (donor ``:519-525``),
            # so the weight rules put its pair on the host and ``shard_segments``
            # moves the weight per device inside the core. The delta's matmul runs
            # where ``x`` does, so the pair moves there first.
            sites = (
                (qkv_a.to(x.device), qkv_b.to(x.device)),
                pair_of(self.in_proj_z),
                pair_of(self.in_proj_b),
                pair_of(self.in_proj_a),
                pair_of(self.out_proj),
                gate.value,
            )
        core = functools.partial(
            _gated_delta_core_with_deltas,
            num_key_heads=self.num_key_heads // num_devices,
            num_value_heads=self.num_value_heads // num_devices,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            ssm_dtype=self.ssm_dtype,
            norm_eps=self.rms_norm_eps,
            output_gate=_OUTPUT_GATE_ACTIVATIONS[self.output_gate_type],
        )
        return F.call_on_mesh(
            core, mesh, out_specs=DeviceMapping(mesh, (Partial(),))
        )(
            x,
            in_proj if self.quantized else (*in_proj, *ba),
            ba if self.quantized else None,
            shard_segments(self.conv1d.weight, self.segments, mesh),
            self.dt_bias,
            self.A_log,
            self.norm.weight,
            self.out_proj.tensors,
            sites,
            state,
            input_row_offsets,
            self.restart,
        )
