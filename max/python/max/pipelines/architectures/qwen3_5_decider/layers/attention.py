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

"""Non-causal full attention for the decider's decision state.

What it installs: the mixer this port puts in the eight ``full_attention``
blocks of the decider's 32-layer hybrid tower (:class:`DeciderAttention`).
``decision_config.json`` declares ``attention_mode: noncausal_full_attention``
and protocol v4 puts one question per ragged row with the answer marker last,
so a query must see every key in its own row, in one prefill, once.

Everything except the mask is MAX's. :class:`DeciderAttention` subclasses
``max.pipelines.architectures.qwen3_5_modulev3.layers.attention.Qwen3_5Attention``
and answers the two arguments that ``forward`` takes for the mask:
``segment_ids``, the label per token, and ``mask_variant``, the mask the chosen
kernel gets. MAX threads that pair at
``max/pipelines/architectures/lfm3/layers/attention.py:116-176``, and this port's
decoder threads it the same way. The projection declarations and their tensor-parallel placement, the per-head ``[Q | gate]``
layout of ``q_proj``, the Q/K RMSNorms that scale by ``1 + weight``, the
partial-RoPE lane reorder, the grouped-query layout the ragged kernels read,
the fused rope that rotates K where the layer just stored it, the sigmoid
output gate, the ``call_on_mesh`` / ``Partial()`` shape of ``forward``, and the
per-device body itself all run as MAX wrote them.

Delta applied, and no other:

- **``MHAMaskVariant.NULL_MASK`` where MAX passes ``CAUSAL_MASK``**: a
  protocol-v4 row is a bidirectional state, so every query attends the whole
  row. Rows stay isolated by ``input_row_offsets``.
- **Under :data:`SEGMENT_MASK_ENABLED`, the cache-less kernel with segment
  labels.** A row that carries the state once and one chunk per question needs
  the state visible to every chunk and each chunk hidden from the others.
  :func:`marker_segments` derives each token's label from the row's own tokens,
  and the decoder threads them to the layer's ``segment_ids`` argument, which
  selects ``flash_attention_ragged_gpu``, the only ragged kernel taking labels.
  That kernel reads no cache, so K is roped in registers inside MAX's body.

MAX's body supplies the projection math, RoPE, cache stores and output gate the
port used to re-declare, and a folded build goes back to the port's body for the
projections. ``FORK_MODEL_NOTES.md`` records what each costs and which runs reach
them.

One more body, and why: :func:`_attention_core_with_deltas` writes the four
adapter deltas of a folded build. MAX's core takes ``q_proj``, ``k_proj`` and
``v_proj`` as one ``stacked_matmul`` over concatenated weights and answers
``F.split``, which hands out three views of that one buffer, so a delta for one
projection has no destination of its own: writing it into the ``q_out`` view
rewrites the ``key`` and ``value`` rows beside it. The port's body adds each
delta into a fresh buffer, and the three operands stop sharing one. The four
sites per block, 32 over the tower, carried no delta before this body. Those 32
pairs hold 12,582,912 adapter elements, 9.7% of the adapter's 129,859,584, and
``FOLD_ALIAS_NOTES.md`` measures what their absence cost in row agreement. ``layers/deltanet.py`` added the same deltas to the
120 linear-attention sites and copies the per-device body the same way.

That reuse has one precondition, and :func:`require_mask_seam` enforces it at
build time: the installed ``Qwen3_5Attention.forward`` has to take the mask
arguments, and the installed ``Qwen3_5TextModel.forward`` has to take
``segment_ids`` and hand them to the layers. A donor that names
``CAUSAL_MASK`` inline in its attention body takes neither, and there the port's
arguments are dead code.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kernels import (
    flash_attention_ragged,
    flash_attention_ragged_gpu,
    fused_qk_ragged_rope,
    rope_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.qwen3_5_modulev3.layers.attention import (
    Qwen3_5Attention,
    _rope_lanes_last,
    rope_ragged_with_position_ids,
)
from max.pipelines.architectures.qwen3_5_modulev3.layers.linear import (
    ProjectionTensors,
    matmul,
    stacked_matmul,
)
from max.pipelines.architectures.qwen3_5_modulev3.layers.norm import (
    rms_norm_scaled,
)
from max.pipelines.architectures.qwen3_5_modulev3.qwen3_5 import (
    Qwen3_5TextModel,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Partial

from .adapter import AdapterGate, pair_of, with_delta
from .deltanet import StateRestart

__all__ = [
    "POSITION_RESTART_ENABLED",
    "SEGMENT_MASK_ENABLED",
    "DeciderAttention",
    "marker_segments",
    "require_mask_seam",
    "restarted_positions",
]

SEGMENT_MASK_ENABLED = os.environ.get("DECIDER_SEGMENT_MASK", "1").lower() in (
    "1",
    "on",
    "true",
)
"""The gate for segment-masked full attention: on unless the environment names
``0``, ``off`` or ``false``.

On, the full-attention layers ask for the segment labels of the step and run
the cache-less kernel. Off, every step answers with ``NULL_MASK`` and the paged
kernel, the route ``mask_off.log`` and ``SQ_NOTES.md`` record. It is read from
the environment once per process, at import, because the mask is a property of
the compiled graph: a step either traces labels into the layers or it does not.
``MASK_NOTES.md`` records what the masked layout costs and answers, and
``LAYOUT_DEFAULT_NOTES.md`` records why the build starts with it on.
"""


def labels_reach_attention() -> bool:
    """True when this process threads the step's labels into the full-attention layers.

    :data:`SEGMENT_MASK_ENABLED` is the only thing that decides it. A step that
    runs the state rewind derives labels for its own segment table and hands
    none of them to the eight ``full_attention`` layers, which stay on the paged
    :func:`flash_attention_ragged` the ``attention kernel:`` line names.
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderTextModel.__init__` prints the
    branch and :meth:`Qwen3_5DeciderDecoder.forward` threads the labels through this
    test, so the log line and the traced kernel come from one rule.
    """
    return SEGMENT_MASK_ENABLED


POSITION_RESTART_ENABLED = os.environ.get(
    "DECIDER_POSITION_RESTART", "1"
).lower() in ("1", "on", "true")
"""The gate for rotary positions that restart at each question chunk: on unless
the environment names ``0``, ``off`` or ``false``.

On, the eight full-attention layers rope at the positions
:func:`restarted_positions` answers. MAX numbers tokens from the ragged row
boundaries, and one state-once row holds every chunk, so under MAX's own rule
chunk 2 counts from the end of chunk 1. It needs
:data:`~.deltanet.STATE_RESTART_ENABLED`, because the positions read that
step's segment table.

Under :data:`SEGMENT_MASK_ENABLED` the layer runs :func:`_attention_core_with_deltas`
even with no adapter bound, because that is the only body this port controls the
key rope in: MAX's cache-less branch ropes K with ``rope_ragged``, whose
``start_pos`` is one number per ragged row
(``max/nn/_graph_ops/__init__.py:3844-3866``), and this core ropes the same key
with ``rope_ragged_with_position_ids``, which takes one uint32 per token
(``max/nn/_graph_ops/__init__.py:4022-4041``, used the same way by
``max/pipelines/architectures/flux2_modulev3/layers/attention.py:89-95``).
``POSITIONS_NOTES.md`` and ``POSITIONS_IN_PORT_NOTES.md`` record what each costs.
"""

MARKER_TOKEN_ID = 248062
"""The ``<|fim_suffix|>`` token id, which ends each question chunk.

``tokenizer.json``'s ``added_tokens`` names it, and
``format/pplx_v13_format.py`` encodes one per chunk, last. Like
:data:`.model_config.LOCATE_MARKER_ID` this is a constant: nothing reads
``tokenizer.json`` at build, so a checkpoint trained on another marker needs it
changed by hand.
"""

SEPARATOR_TOKEN_ID = 248061
"""The ``<|fim_middle|>`` token id, which closes the state block.

The renderer puts it once per row, between the state and the first question
chunk, so it is the last token of the segment every chunk may see.
"""


def marker_segments(
    tokens: Tensor,
    input_row_offsets: Tensor,
    marker_id: int = MARKER_TOKEN_ID,
    separator_id: int = SEPARATOR_TOKEN_ID,
) -> Tensor:
    """Returns each token's int32 segment id, read off the step's tokens.

    The state block and the separator that closes it get -1, which the segment
    mask makes visible to every token of the row. Chunk ``i`` -- the question
    text and the marker that ends it -- gets ``i``, so a marker reads its own
    chunk and the state, and reads no other chunk. The two running counts are
    global over the ragged batch, so each one is offset by the count its row
    starts with: labels restart at every row and never go down inside one,
    which is what the kernel asks for.
    """
    row_lengths = F.cast(
        input_row_offsets[1:] - input_row_offsets[:-1], DType.int32
    )
    row_ends = F.cast(input_row_offsets[1:] - 1, DType.int64)

    def before_each_row(flags: Tensor) -> tuple[Tensor, Tensor]:
        """Each token's own count, and the count its row starts with."""
        counted = F.cast(flags, DType.int32)
        before = F.cumsum(counted, axis=0, exclusive=True)
        per_row = F.gather(F.cumsum(counted, axis=0), row_ends, axis=0)
        base = F.cumsum(per_row, axis=0) - per_row
        return before, F.repeat_interleave(
            base, row_lengths, out_dim=counted.shape[0]
        )

    markers, marker_base = before_each_row(tokens == marker_id)
    separators, separator_base = before_each_row(tokens == separator_id)
    # A token of the state block has met no marker, so its chunk count is 0;
    # subtracting the separator test turns that 0 into the shared -1.
    chunk = markers - marker_base
    shared = F.cast(separators == separator_base, DType.int32)
    return chunk - shared


def restarted_positions(edges: Tensor, segment_index: Tensor) -> Tensor:
    """Returns each token's rope position, restarted at the end of the state.

    ``edges`` is the step's ``[blocks + 2]`` int32 segment table
    (:func:`~.deltanet.restart_edges`) and ``segment_index`` is each token's
    segment index (:func:`~.deltanet.restart_segment_index`). A token of
    segment ``s``
    gets ``token_index - edges[s] + edges[1]``: the count of tokens its own
    chunk ran through, offset by the state's length, so chunk 2 starts where the
    state ends and every chunk counts its own tokens from 0. Segment 0 gathers
    ``edges[0]``, which is 0, and keeps its 0-based positions.

    Returns:
        ``[1, total_seq_len]`` uint32, the shape and dtype MAX's rope positions
        input takes
        (``max/nn/_graph_ops/__init__.py:2717-2719``, and the donor's M-RoPE
        ``freq_row_ids`` at
        ``max/pipelines/architectures/qwen3_5_modulev3/qwen3_5.py:365-370``).
    """
    token_index = F.range(
        0,
        segment_index.shape[0],
        1,
        out_dim=segment_index.shape[0],
        dtype=DType.int32,
        device=segment_index.device,
    )
    # ``edges[0]`` for the state block, ``edges[1]`` (the state's length) for
    # every chunk, read as the edge before the segment each token is in.
    seg_start = F.gather(edges, segment_index, axis=0)
    state_offset = F.gather(
        edges, F.cast(F.where(F.equal(seg_start, 0), 0, 1), DType.int32), axis=0
    )
    positions = token_index - seg_start + state_offset
    return F.cast(positions, DType.uint32).unsqueeze(0)


def require_mask_seam() -> None:
    """Refuses a build on a MAX that cannot take this port's mask arguments.

    Two donor seams carry the mask. ``Qwen3_5Attention.forward`` answers
    ``mask_variant`` and ``segment_ids``, and ``Qwen3_5TextModel.forward`` takes
    ``segment_ids`` and threads it to the eight ``full_attention`` layers. Stock
    MAX declares neither: its attention body names
    ``MHAMaskVariant.CAUSAL_MASK`` inline, and its text model has no
    ``segment_ids`` parameter. A build there traces a causal row under a
    protocol-v4 prompt whose marker is last, and the port's ``attention kernel:``
    line still reports ``NULL_MASK``. An install carrying only the attention half
    fails inside ``Qwen3_5TextModel.forward``, and the ``TypeError`` names
    ``segment_ids``. Both refusals belong before the first layer is built, and
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderTextModel.__init__` runs this one
    first.

    Raises:
        RuntimeError: When a seam lacks an argument the port calls it with. The
            message names each missing argument, the file the argument was looked
            for in, and the patch that adds it.
    """
    seams = (
        (
            "Qwen3_5Attention.forward",
            Qwen3_5Attention.forward,
            ("freq_row_ids", "segment_ids", "mask_variant"),
        ),
        (
            "Qwen3_5TextModel.forward",
            Qwen3_5TextModel.forward,
            ("segment_ids",),
        ),
    )
    gaps: list[str] = []
    for label, method, needed in seams:
        parameters = inspect.signature(method).parameters
        missing = [name for name in needed if name not in parameters]
        if missing:
            source = getattr(inspect.getmodule(method), "__file__", "unknown")
            gaps.append(
                f"{label}, read from {source}, declares"
                f" ({', '.join(parameters)}) and takes no {', '.join(missing)}"
            )
    if not gaps:
        return
    raise RuntimeError(
        "this MAX has no mask seam, and the port refuses to serve a graph whose"
        " mask it cannot set. The port answers"
        " Qwen3_5Attention.forward's mask_variant and segment_ids with"
        " MHAMaskVariant.NULL_MASK and the step's labels, and"
        " Qwen3_5DeciderDecoder.forward passes segment_ids to"
        " Qwen3_5TextModel.forward, which threads it to the eight"
        " full_attention layers. Missing:"
        + "".join(f"\n  - {gap}" for gap in gaps)
        + "\nBoth halves ship with this package's tree:"
        " max/pipelines/architectures/qwen3_5_modulev3/layers/attention.py"
        " carries the arguments, and"
        " max/pipelines/architectures/qwen3_5_modulev3/qwen3_5.py threads the"
        " segment_ids channel. Reaching this error means the decider and the"
        " Qwen3.5 build it subclasses disagree in version, which an"
        " out-of-date install of one of the two causes."
    )


AttentionSites = tuple[
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[Tensor, Tensor],
    Tensor,
]
"""The five delta slots :func:`_attention_core_with_deltas` reads, in order.

``(q_proj, k_proj, v_proj, o_proj, gate)``: each pair is that site's
``(lora_a, lora_b)``, and ``gate`` is the step's per-token adapter gate. The
fold binds all five on a layer or none of them, so the slots are complete or
the whole tuple is ``None``.
"""


def _attention_core_with_deltas(
    x: Tensor,
    qkv: Sequence[ProjectionTensors],
    q_norm: Tensor,
    k_norm: Tensor,
    o_proj: ProjectionTensors,
    adapter: AttentionSites | None,
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
    positions: Tensor | None = None,
) -> Tensor:
    """Runs one device's heads and returns its ``o_proj`` partial sums, deltas on.

    Structurally MAX's ``_attention_core``, with three additions: each of the
    four sites' adapter delta is added to that site's rows, and each add names a
    fresh destination because ``F.split`` hands out three views of one
    ``stacked_matmul`` buffer. Writing a ``q_proj`` delta into the ``q_out`` view
    would rewrite ``key`` and ``value`` beside it, so the delta lands in its own
    buffer and the three operands stop sharing one. The rope, the Q/K norms, the
    cache stores, the two attention kernels and the sigmoid output gate are the
    donor's calls in the donor's order; the base term stays one matmul, which is
    row-for-row the three base projections.

    Args:
        adapter: the five slots ``(q_pair, k_pair, v_pair, o_pair, gate)`` of
            a folded build (:data:`AttentionSites`), each pair the site's
            ``(lora_a, lora_b)`` and ``gate`` the step's per-token gate, or
            ``None`` on a build with no adapter bound, which leaves every
            projection at its base term. A half-bound set is not a thing the
            fold produces.
        positions: ``[1, total_seq_len]`` uint32 rope position per token
            (:func:`restarted_positions`), or ``None`` for MAX's own numbering.
            The label branch reads it for the key rope; the paged branch already
            takes per-token positions through ``freq_row_ids``, which the caller
            fills from the same tensor.

    Raises:
        NotImplementedError: When the KV cache is FP8. That branch ropes and
            stores through ``rope_split_store_ragged``, which takes one flat
            ``[Q | K | V]`` operand, and a delta has nowhere to go in a buffer
            the three projections already share.
    """
    total_seq_len = x.shape[0]
    q_dim = int(qkv[0].weight.shape[0])
    kv_dim = int(qkv[1].weight.shape[0])
    n_heads = q_dim // (2 * head_dim)
    n_kv_heads = kv_dim // head_dim
    norm = functools.partial(rms_norm_scaled, eps=norm_eps)

    q_out, key, value = F.split(
        stacked_matmul(x, qkv), [q_dim, kv_dim, kv_dim], axis=-1
    )
    if adapter is not None:
        q_pair, k_pair, v_pair, o_pair, adapter_gate = adapter
        q_out = with_delta(q_out, x, q_pair, adapter_gate)
        key = with_delta(key, x, k_pair, adapter_gate)
        value = with_delta(value, x, v_pair, adapter_gate)
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
        raise NotImplementedError(
            "the folded full-attention body has no FP8 KV path:"
            " rope_split_store_ragged ropes and stores from one flat"
            " [Q | K | V] operand, and the four deltas need operands of their"
            " own. Run an FP8 cache with the fold off, or add the deltas to"
            " that kernel's three outputs."
        )
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
        if kv_params.enable_prefix_caching:
            raise ValueError(
                "segment_ids run through flash_attention_ragged_gpu, which"
                " reads no KV cache, so the layer can only rope a row this"
                " step wrote whole; enable_prefix_caching lets a step resume"
                " pages an earlier step wrote, and rope_ragged would number"
                " that row's tokens past the cached length while the kernel"
                " reads no cached keys"
            )
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
        if positions is not None:
            # The per-token rope answers a TensorValue, which has no mapping to
            # rebind; the donor's rope_ragged answers a Tensor whose mapping is
            # rebound to the key's so the sharded operands line up.
            key_rows = rope_ragged_with_position_ids(
                key, freqs_cis.cast(query.dtype), positions, interleaved=True
            )
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
    # ``o_proj`` reads the gated attention output, so its delta multiplies that
    # same operand ``QuantizedLinear.forward`` would have been handed.
    o_in = attn_out * F.sigmoid(gate)
    base_out = matmul(o_in, o_proj)
    if adapter is None:
        return base_out
    *_, o_pair, adapter_gate = adapter
    return with_delta(base_out, o_in, o_pair, adapter_gate)


class DeciderAttention(Qwen3_5Attention):
    """Qwen3.5 gated full attention that attends the row, or the row's segments.

    ``__init__`` and ``forward`` are MAX's, so the layer declares the same
    ``q_proj``/``k_proj``/``v_proj`` head-row split, ``o_proj`` input split and
    per-head Q/K norms as the block mixer it replaces, and loads the same
    checkpoint tensors. Only the mask arguments change.

    ``segment_ids`` arrives as an argument of ``forward``, threaded per step by
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderDecoder.forward` under
    :data:`SEGMENT_MASK_ENABLED`, and this class answers ``mask_variant`` with
    ``NULL_MASK``. ``restart`` is that graph's segment table, put on the layer
    by :meth:`qwen3_5_decider.decider.Qwen3_5DeciderTextModel.__init__` when
    :data:`POSITION_RESTART_ENABLED` is set, and is what :meth:`forward` reads
    for the step's rope positions.
    """

    restart: StateRestart | None = None
    """The step's :class:`~.deltanet.StateRestart`, or ``None``.

    A layer the decoder never wired keeps ``None`` and ropes at MAX's own
    positions. Only :meth:`forward` reads it, for ``restart.positions``, and
    only when :data:`POSITION_RESTART_ENABLED` traced them.
    """

    gate: AdapterGate
    """The step's :class:`~.adapter.AdapterGate`, bound by ``install`` on a
    folded build and absent on a decisions build. Its presence is what picks
    the body: MAX's for the plain projections, the port's for the four that
    carry a delta. :func:`~.adapter._check_replicated` refuses a
    tensor-parallel build, so the pairs reach one device.
    """

    def _deltas_core(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        freq_row_ids: Tensor | None,
        segment_ids: Tensor | None,
        adapter_gate: Tensor | None,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Runs this layer through :func:`_attention_core_with_deltas`.

        The same ``call_on_mesh`` / ``Partial()`` shape MAX's ``forward`` uses,
        so the row-parallel ``o_proj`` leaves partial sums for the block's
        residual add to reduce.
        """
        core = functools.partial(
            _attention_core_with_deltas,
            kv_params=self.kv_params,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            scale=self.scale,
            norm_eps=self.norm_eps,
            segment_ids=segment_ids,
            mask_variant=MHAMaskVariant.NULL_MASK,
            positions=positions,
        )
        # A gate arrives with the four pairs bound on this layer; without one
        # the core reads no pair and ropes only.
        sites: AttentionSites | None = (
            None
            if adapter_gate is None
            else (
                pair_of(self.q_proj),
                pair_of(self.k_proj),
                pair_of(self.v_proj),
                pair_of(self.o_proj),
                adapter_gate,
            )
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
            sites,
            kv_collection,
            layer_idx,
            freqs_cis,
            input_row_offsets,
            freq_row_ids,
        )

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
        """MAX's attention body, answered non-causally and roped under the gate.

        ``mask_variant`` is taken from the donor's signature and ignored: the
        protocol-v4 row is one bidirectional state, and the body always asks
        ``NULL_MASK``, which masks nothing inside the row range the kernel
        already walks, so rows stay apart and every query sees its whole row.
        ``segment_ids`` is the step's label per token, from
        :meth:`qwen3_5_decider.decider.Qwen3_5DeciderDecoder.forward`.

        ``freq_row_ids`` is MAX's channel for per-token rope positions: with it
        set, ``fused_qk_ragged_rope`` reads each token's position from that row
        for the arithmetic that would otherwise compute ``cache_length +
        token_idx``
        (``max/pipelines/architectures/qwen3_5_modulev3/layers/attention.py:180-188``
        and its ``position_ids`` check at
        ``max/nn/_graph_ops/__init__.py:2717-2719``). The donor fills it only
        for M-RoPE checkpoints, and protocol v4 text has none, so this is where
        the restarted positions enter the graph.

        Raises:
            RuntimeError: With :data:`SEGMENT_MASK_ENABLED` set and no labels
                threaded to this call, which means the step ran no
                :func:`marker_segments`.
        """
        positions: Tensor | None = None
        if self.restart is not None and self.restart.positions is not None:
            freq_row_ids = positions = self.restart.positions
        if labels_reach_attention() and segment_ids is None:
            raise RuntimeError(
                "SEGMENT_MASK_ENABLED is set and this layer has no segment"
                " labels to trace, so the step would run unmasked attention"
                " under a gate that says otherwise."
                " Qwen3_5DeciderDecoder.forward runs marker_segments(tokens,"
                " input_row_offsets) and threads the result to the text"
                " model's segment_ids argument"
            )
        if not labels_reach_attention() and segment_ids is not None:
            raise RuntimeError(
                "this layer was handed segment labels with"
                " DECIDER_SEGMENT_MASK switched off, so it would trace"
                " flash_attention_ragged_gpu with per-token labels while"
                " Qwen3_5DeciderTextModel.__init__ prints"
                " 'attention kernel: flash_attention_ragged under NULL_MASK'."
                " labels_reach_attention() answers for both: thread the labels"
                " under the mask gate, or call this layer without them."
                " The step's segment table for DECIDER_STATE_RESTART comes from"
                " restart_edges(labels, blocks), not from this argument"
            )
        gate = getattr(self, "gate", None)
        if gate is None and labels_reach_attention() and positions is not None:
            # MAX's own body would rope Q at the restarted numbers and K at one
            # start position per ragged row, and the two operands would disagree
            # about where a token sits. The port's body is the one that takes
            # per-token positions on both operands, so the masked restart runs
            # there with no adapter gate: the projections keep their base terms.
            return self._deltas_core(
                x,
                kv_collection,
                layer_idx,
                freqs_cis,
                input_row_offsets,
                freq_row_ids,
                segment_ids,
                None,
                positions,
            )
        if gate is not None:
            # A folded build put ``(lora_a, lora_b)`` on this layer's four
            # projections, and MAX's body reads those four as bare
            # ``ProjectionTensors``, so no wrapped ``forward`` runs their
            # deltas. The port's body takes each projection's destination and
            # writes it. A decisions build binds no gate, and the layer keeps
            # MAX's body: same graph, same probabilities, one less copy.
            return self._deltas_core(
                x,
                kv_collection,
                layer_idx,
                freqs_cis,
                input_row_offsets,
                freq_row_ids,
                segment_ids,
                gate.value,
                positions,
            )
        return super().forward(
            x,
            kv_collection,
            layer_idx,
            freqs_cis,
            input_row_offsets,
            # Keywords from this point down: the donor's own order is
            # ``freq_row_ids, segment_ids, mask_variant``, and a donor that
            # reorders them would hand ``segment_ids`` to ``mask_variant``, and
            # the layer would trace a mask this port did not ask for.
            freq_row_ids=freq_row_ids,
            segment_ids=segment_ids,
            mask_variant=MHAMaskVariant.NULL_MASK,
        )
