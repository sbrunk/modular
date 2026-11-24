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

"""The decider's registered decoder: the Qwen3.5 hybrid tower, readout head.

This is the ModuleV3 port of the graph-API spike in ``qwen3_5_decider/``. One
protocol-v4 batch goes in (ragged rows, one per question, each ending at the
answer marker) and one row of 255 readout scores comes out per row, which is
what MAX's ``LabelScorer`` gathers the option probabilities from.

Structurally mirrors
``max.pipelines.architectures.qwen3_5_modulev3.qwen3_5``: the registered
``DECODER`` unflattens the cache tree into the attention pages and the
gated-DeltaNet state leaves, and the text model is the donor's
:class:`Qwen3_5TextModel` -- embedding, hybrid layer schedule, final norm,
last-token logits. The donor's own helpers do that plumbing
(:func:`replicate`, :func:`distributed_state`, :data:`ATTN_CACHE_KEY`,
:data:`STATE_CACHE_KEY`), and the gated-DeltaNet mixer is the donor's
:class:`GatedDeltaNet` unmodified.

Deltas layered on top of the donor:

- **The head is the 255-row readout, in float32.** ``lm_head`` stays absent, as
  the donor leaves it for a ``tie_word_embeddings`` checkpoint, and the
  tied-embedding matmul it would fall back to is replaced by ``readout``. The
  matmul runs in float32 on the bfloat16 backbone, which is what the reference
  does on a float32 run and what the graph-API route measured against.
- **Full attention is non-causal.** Each ``full_attention`` layer's mixer is
  replaced by :class:`~.layers.attention.DeciderAttention`, which answers MAX's
  ``mask_variant`` argument with ``NULL_MASK``. A MAX whose
  ``Qwen3_5Attention.forward`` takes no mask argument is refused at build by
  :func:`~.layers.attention.require_mask_seam`: that donor's attention call
  reads its own inline mask and ignores the answer. Under
  :data:`~.layers.attention.SEGMENT_MASK_ENABLED` the step's segment labels go
  into the ``segment_ids`` argument beside it, which sends those eight layers
  through the cache-less kernel. Under
  :data:`~.layers.deltanet.STATE_RESTART_ENABLED` the labels are read off the
  step's span edges, which arrive as a host-filled graph input and say where the
  state ends and where each question ends; without that input,
  :func:`~.layers.attention.marker_segments` counts the ``<|fim_suffix|>`` token
  that closes each chunk.
- **No vision merge, no M-RoPE positions, no speculative path.** The config
  clears the tower and the M-RoPE section, so the rotary table is the static
  one indexed by position within a row, and the graph takes no
  ``token_positions`` input.
- **The weight-coverage audit runs at build.** Every tensor of the adapted
  checkpoint that matches no parameter is reported; the ones this milestone
  leaves out are named, counted, and everything else raises.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Iterable
from typing import Any, ClassVar

from max import functional as F
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.kv_cache import (
    PagedCacheValues,
    kv_input_values,
)
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    RecurrentStateInputsPerDevice,
)
from max.nn.linear import Linear
from max.pipelines.architectures.qwen3_5_modulev3.qwen3_5 import (
    FULL_ATTENTION,
    Qwen3_5TextModel,
    distributed_state,
    replicate,
)
from max.pipelines.architectures.qwen3_5_modulev3.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
)
from max.pipelines.lib.role_contracts import (
    DecoderInputs,
    DecoderOutputs,
)
from max.tensor import Tensor, default_dtype

from .layers.adapter import AdapterSites, install, token_gate
from .layers.attention import (
    MARKER_TOKEN_ID,
    POSITION_RESTART_ENABLED,
    SEGMENT_MASK_ENABLED,
    DeciderAttention,
    labels_reach_attention,
    marker_segments,
    require_mask_seam,
    restarted_positions,
)
from .layers.deltanet import (
    RESTART_BLOCKS,
    STATE_RESTART_ENABLED,
    LocateGatedDeltaNet,
    StateRestart,
    attention_labels,
    segment_index_from_edges,
)
from .model_config import Qwen3_5DeciderConfig

__all__ = ["Qwen3_5DeciderDecoder", "Qwen3_5DeciderTextModel"]

logger = logging.getLogger(__name__)

_LEFT_OUT_PREFIXES: tuple[str, ...] = ("visual.",)
"""The vision tower: this milestone registers no ``VISION`` module."""

_LEFT_OUT_SUFFIXES: tuple[str, ...] = (".lora_a", ".lora_b", ".locate")
"""The unmerged Locate adapter, excused while :data:`ADAPTER_FOLD_ENABLED` is
False. Flipping that flag binds the tensors as parameters and this excuse stops
matching, so a rename in :mod:`.weight_adapters` that strands one of them fails
the audit."""

ADAPTER_FOLD_ENABLED = os.environ.get("DECIDER_ADAPTER_FOLD", "1").lower() in (
    "1",
    "true",
    "on",
)
"""The gate for folding the Locate adapter into this graph, on unless the
environment names ``0``, ``off`` or ``false``. A checkpoint with no
``*.lora_a`` tensors has nothing to fold and builds no sites whatever this
says.

:mod:`.layers.adapter` implements the fold: 248 ``(lora_a, lora_b)`` pairs,
the marker row, the delta ops of ``LoRALinear.forward``, and a per-token gate
:meth:`Qwen3_5DeciderDecoder.forward` derives in the graph from the
``<|locate|>`` token, so a decisions row carries a gate of exact 0.0 and
:func:`~.layers.adapter.with_delta` returns its base bit for bit. A folded
build traces the donor's blocks inline (``use_subgraphs = False``), the route
a speculative verify takes for the same reason: MAX refuses to move a tensor
built in the enclosing graph -- the gate is one -- into a layer subgraph.
Every mixer family the tower uses has a body that reads the gate:
:class:`~.layers.attention.DeciderAttention`, :class:`~.layers.mlp.LocateMLP`
and :class:`~.layers.deltanet.LocateGatedDeltaNet`, and
:func:`~.layers.adapter.install` refuses a family whose sites all lack one.
Off, the pairs load, the audit excuses them by suffix, and every step runs
the plain backbone.

A folded build costs the adapter's float32 tensors in device memory: on one
RTX 4090 at ``--device-memory-utilization 0.6`` the KV cache shrinks from
1.97 GiB to 828 MiB, about 1.2 GiB for the 497 tensors, and the step pays
248 rank-64 delta pairs whose gate is an exact zero on a decisions row.
"""

_LEFT_OUT_NAMES: frozenset[str] = frozenset(
    {
        "start_bias",
        "end_bias",
        "width_bias.weight",
        "thresholds.weight",
        "thresholds.bias",
        *(
            f"{proj}.{part}"
            for proj in ("qe", "qp", "qs", "te", "tp", "ts")
            for part in ("weight", "bias")
        ),
    }
)
"""The Locate span head's projections, biases and thresholds."""


def _left_out_reason(name: str) -> str | None:
    """Returns why a checkpoint tensor is unused here, or ``None`` if not.

    The two groups are the milestone list, not a wildcard: the vision tower
    (no ``VISION`` module is registered) and the Locate span head, which the
    decoder has no consumer for until the span scoring graph exists.
    """
    if name.startswith(_LEFT_OUT_PREFIXES):
        return "vision tower"
    if name.endswith(_LEFT_OUT_SUFFIXES):
        return "unmerged Locate adapter"
    if name in _LEFT_OUT_NAMES:
        return "Locate span head"
    return None


def audit_weight_coverage(
    *, checkpoint_tensor_names: Iterable[str], module: Module[..., Any]
) -> None:
    """Raises if a checkpoint tensor matches no parameter of ``module``.

    ``compile(weights=...)`` checks the state dict from the parameter side
    only: a tensor that no parameter reads is dropped without a warning, so a
    stale rename or a component the module never builds shows up as accuracy
    drift, not as an error. Only the names this milestone documents as unused
    are excused, and they are counted in the log so a change in that set is
    visible.

    Args:
        checkpoint_tensor_names: The adapted checkpoint's tensor names, as the
            config recorded them when it finalized.
        module: The lazily built root module, whose parameter paths are the
            names ``compile`` resolves.

    Raises:
        RuntimeError: If any tensor matches no parameter and is not excused.
    """
    names = frozenset(checkpoint_tensor_names)
    consumed = {name for name, _ in module.parameters}
    unconsumed = sorted(names - consumed)
    reasons = {name: _left_out_reason(name) for name in unconsumed}
    unexpected = sorted(
        name for name, reason in reasons.items() if reason is None
    )
    if unexpected:
        sample = unexpected[:8]
        raise RuntimeError(
            f"state-dict audit: {len(unexpected)} checkpoint tensor(s) match"
            " no module parameter. Likely cause: a stale rename in"
            " weight_adapters.py or a component the module does not build."
            f" First {len(sample)}: {sample}"
        )
    counts: Counter[str] = Counter(
        reason for reason in reasons.values() if reason is not None
    )
    logger.info(
        "state-dict audit: %d checkpoint tensors, %d consumed by the decoder,"
        " %d left unused on purpose: %s.",
        len(names),
        len(names & consumed),
        len(unconsumed),
        ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
        or "none",
    )


class Qwen3_5DeciderTextModel(Qwen3_5TextModel):
    """The donor's text model with the readout head and non-causal attention.

    ``Qwen3_5TextModel`` builds the embedding, the 32-block hybrid schedule,
    the final norm and the head, and its ``hidden_states``/``logits``/``forward``
    do the rest. Two things are re-declared here: the mixer of each
    full-attention block, and the head.
    """

    def __init__(self, config: Qwen3_5DeciderConfig) -> None:
        """Builds the donor stack, then swaps in this port's two pieces.

        The blocks and the projections the swap does not touch are the donor's
        instances, kept as built. The eight full-attention mixers are rebuilt to
        answer the mask seam another way; the head is re-declared below.

        Raises:
            RuntimeError: When the gate combination cannot serve the layout it
                names: ``DECIDER_POSITION_RESTART`` on while
                ``DECIDER_STATE_RESTART`` is off, or ``DECIDER_SEGMENT_MASK`` on
                while ``DECIDER_STATE_RESTART`` is off. Also when the loaded MAX
                declares no mask seam; see
                :func:`~.layers.attention.require_mask_seam`.
        """
        # Before anything is built: on a donor that does not ask its layer for the
        # mask arguments, the swap below would trace CAUSAL_MASK and report
        # NULL_MASK.
        require_mask_seam()
        if POSITION_RESTART_ENABLED and not STATE_RESTART_ENABLED:
            raise RuntimeError(
                "DECIDER_POSITION_RESTART is on while DECIDER_STATE_RESTART is"
                " off. The restarted positions read the step's"
                " segment table, which only the restart gate declares, so the"
                " rope would have no edges to count positions from"
            )
        if SEGMENT_MASK_ENABLED and not STATE_RESTART_ENABLED:
            raise RuntimeError(
                "DECIDER_SEGMENT_MASK is on while DECIDER_STATE_RESTART is"
                " off, and this build refuses the pair."
                " Qwen3_5DeciderDecoder.token_position_type() declares the"
                " token_positions input under the restart gate alone, so with"
                " the gate off the request's own span counts"
                " (qwen3_5_decider.batch_processor.SegmentBounds) have no"
                " graph input to arrive in. Qwen3_5DeciderDecoder.forward then hands"
                " the layers marker_segments(tokens, input_row_offsets), which"
                f" counts token {MARKER_TOKEN_ID} per chunk; a Locate row holds"
                " none of that token, so one label covers the question area and"
                " the mask cuts nothing. Set DECIDER_STATE_RESTART=1 with"
                " DECIDER_SEGMENT_MASK=1, or set DECIDER_SEGMENT_MASK=0."
            )
        # DECIDER_POSITION_RESTART with DECIDER_SEGMENT_MASK used to raise:
        # MAX's masked branch ropes K with rope_ragged, one start position per
        # ragged row, so per-token positions could only restart Q.
        # DeciderAttention.forward answers that pair through the port's own core,
        # which ropes Q through fused_qk_ragged_rope(position_ids=...) and K
        # through rope_ragged_with_position_ids, both per token.
        super().__init__(config)
        # A checkpoint that ships the Locate span head gets the ``label_hidden``
        # channel: the route that answers Locate reads the gathered rows
        # themselves, and the only affordable place to gather them is the pass
        # that already computed them. A checkpoint without the head stays a
        # plain two-channel graph.
        if not config.checkpoint_tensor_names.isdisjoint(_LEFT_OUT_NAMES):
            self.label_hidden_declared = True
        # The donor's block builds each mixer with the checkpoint's quantisation
        # scheme for that layer: ``scheme`` and ``attn_quant`` at
        # ``qwen3_5.py:143-144``, ``quant_config=attn_quant`` at
        # ``qwen3_5.py:162`` for attention and ``qwen3_5.py:177`` for the
        # DeltaNet. A swap that leaves that argument out declares unquantised
        # projections over a quantised checkpoint, so both swaps below read this
        # scheme.
        scheme = config.quant_scheme
        for index, kind in enumerate(self.layer_types):
            if kind == FULL_ATTENTION:
                self.layers[index].self_attn = DeciderAttention(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.attn_kv_params.head_dim,
                    rotary_dim=self.rotary_dim,
                    kv_params=config.attn_kv_params,
                    layer_idx=index,
                    dtype=config.compute_dtype,
                    norm_dtype=config.norm_dtype or config.compute_dtype,
                    norm_eps=config.rms_norm_eps or 1e-6,
                    scale=config.attention_multiplier,
                    quant_config=scheme.attn_config(index) if scheme else None,
                )
        # Labels are not kept on the model: :meth:`Qwen3_5DeciderDecoder.forward`
        # derives them per step and threads them to the eight full-attention
        # layers through the text model's ``segment_ids`` argument, and
        # :func:`~.layers.attention.labels_reach_attention` decides that
        # hand-off. With the mask gate off those layers get no labels and keep
        # answering ``NULL_MASK`` to the paged kernel, even in a step that
        # derives labels for the rewind's segment table. The restarted DeltaNet
        # reads the same labels, so a build that only rewinds the state still
        # derives them.
        # One segment table per compiled graph for the 24 linear-attention
        # mixers, bound once per step below: the end of the state, then one edge
        # per question block. Off, every layer runs its step in one call.
        self.restart: StateRestart | None = None
        if STATE_RESTART_ENABLED:
            self.restart = StateRestart(blocks=RESTART_BLOCKS)
            for index, kind in enumerate(self.layer_types):
                if kind == FULL_ATTENTION:
                    continue
                block = self.layers[index]
                donor = block.linear_attn
                assert donor is not None, "a linear block has a DeltaNet mixer"
                # Same class, same ``__init__``, same constructor arguments as
                # the donor block built (``qwen3_5.py:167-178``), so every
                # checkpoint tensor still lands on the same parameter path. The
                # port's subclass is what has a segment table to read.
                block.linear_attn = LocateGatedDeltaNet(
                    hidden_size=config.hidden_size,
                    num_key_heads=config.linear_num_key_heads,
                    num_value_heads=config.linear_num_value_heads,
                    key_head_dim=config.linear_key_head_dim,
                    value_head_dim=config.linear_value_head_dim,
                    conv_kernel_size=config.linear_conv_kernel_dim,
                    dtype=config.compute_dtype,
                    rms_norm_eps=config.rms_norm_eps or 1e-6,
                    ssm_dtype=config.mamba_ssm_dtype,
                    quant_config=(
                        scheme.attn_config(index) if scheme else None
                    ),
                    scalar_dtypes=config.linear_attn_scalar_dtypes,
                )
                block.linear_attn.restart = self.restart
            if POSITION_RESTART_ENABLED:
                # The same table, read by the full-attention mixers: their
                # ``forward`` answers MAX's ``freq_row_ids`` argument with the
                # positions this graph binds per step.
                for index, kind in enumerate(self.layer_types):
                    if kind != FULL_ATTENTION:
                        continue
                    attention = self.layers[index].self_attn
                    # ``__init__`` replaced every full-attention mixer with a
                    # DeciderAttention above; MAX's block types the attribute
                    # as its own class.
                    assert isinstance(attention, DeciderAttention)
                    attention.restart = self.restart
        if SEGMENT_MASK_ENABLED or STATE_RESTART_ENABLED:
            # The labels and the segment table are built in this graph and read
            # inside a layer, and MAX refuses to move an enclosing-graph tensor
            # into one of the donor's layer subgraphs, so the blocks trace
            # inline: the same move a folded adapter build and a speculative
            # verify make.
            self.use_subgraphs = False
        kernel = (
            "flash_attention_ragged_gpu with per-token segment labels"
            if labels_reach_attention()
            else "flash_attention_ragged under NULL_MASK"
        )
        full_layers = sum(kind == FULL_ATTENTION for kind in self.layer_types)
        print(
            f"attention kernel: {kernel} over {full_layers} of"
            f" {len(self.layer_types)} layers",
            flush=True,
        )
        linear_layers = len(self.layer_types) - full_layers
        position_mode = (
            f"restarted at the end of the state for each of {RESTART_BLOCKS}"
            " question blocks"
            if POSITION_RESTART_ENABLED
            else "continuous across a row, from the row boundaries"
        )
        print(
            f"rope positions: {position_mode} over {full_layers}"
            " full-attention layers",
            flush=True,
        )
        print(
            f"delta state layout: {
                'snapshot and rewind before each of'
                f' {RESTART_BLOCKS} question blocks'
                if STATE_RESTART_ENABLED
                else 'one segment per row, no rewind'
            } over {linear_layers}"
            f" linear-attention layers"
            + (
                f"; a step stating more than {RESTART_BLOCKS + 1} spans is"
                " refused at the batch processor"
                if STATE_RESTART_ENABLED
                else ""
            ),
            flush=True,
        )
        boundary_rule = (
            "the step's [blocks + 2] span edges, filled on the host by"
            " Qwen3_5DeciderBatchProcessor.token_positions"
            if STATE_RESTART_ENABLED
            else f"token {MARKER_TOKEN_ID} counted inside the graph"
        )
        print(
            f"segment boundaries from: {boundary_rule}",
            flush=True,
        )
        # Which row layout a caller should render, and the gates that decide it.
        # The choice belongs to the caller: ``format/pplx_v13_format.py``
        # ``encode`` writes one row per question, ``encode_state_once`` writes
        # one row with the state once, and ``encode_default`` picks between them
        # by question count. The three gates decide whether a shared row
        # answers: with any of them off, a row holding two chunks reads the
        # other chunk's state and continuous rope numbers, the wrong answer,
        # asked and refused at build time.
        off_gates = [
            name
            for name, on in (
                ("DECIDER_SEGMENT_MASK", labels_reach_attention()),
                ("DECIDER_STATE_RESTART", STATE_RESTART_ENABLED),
                ("DECIDER_POSITION_RESTART", POSITION_RESTART_ENABLED),
            )
            if not on
        ]
        layout_rule = (
            "a request naming 2 to"
            f" {RESTART_BLOCKS} questions reads one row: the state once, then"
            " one chunk per question. A request naming one question keeps its"
            " own row, the layout every published score was measured with"
            if not off_gates
            else "one row per question: "
            + ", ".join(off_gates)
            + " is off, so a row holding two chunks would answer against the"
            " other chunk's state"
        )
        print(f"request layout: {layout_rule}", flush=True)
        # The decision head: ``readout.safetensors``' one BF16 tensor, shaped
        # [codes, hidden] the way a projection is stored. Not tensor-parallel:
        # this port serves one device, and the readout's 255 rows are not split.
        with default_dtype(config.compute_dtype):
            self.readout = Linear(
                config.hidden_size, config.readout_rows, bias=False
            )
        # The Locate adapter, last, so every projection the donor stack and this
        # file's two swaps built is already in the tree. ``readout`` is skipped: the
        # adapter targets the 248 projections of ``backbone.language_model``,
        # and this head is the graph's.
        self.adapter: AdapterSites | None = None
        if ADAPTER_FOLD_ENABLED and config.adapter_rank:
            # The donor traces each block through ``as_subgraph``, and MAX
            # refuses to move a tensor built in the enclosing graph into one:
            # ``Tensor.__tensorvalue__`` calls ``_sync_realize`` when the
            # tensor's graph is not the current one, and a graph context raises
            # ``TypeError: Can't realize from a graph context.``
            # (``max/tensor/realization_context.py:683``). The gate is such a
            # tensor, so a folded build traces its blocks inline; a speculative
            # verify sets ``use_subgraphs`` off for the same reason
            # (``max/pipelines/architectures/qwen3_5_modulev3/qwen3_5.py:346``).
            self.use_subgraphs = False
            self.adapter = install(
                self,
                rank=config.adapter_rank,
                marker_id=config.locate_marker_id,
                skip=("readout", "embed_tokens"),
            )
            logger.info(
                "Locate adapter bound at rank %d over %d projections plus the"
                " input embedding; marker id %d gates it per row.",
                config.adapter_rank,
                self.adapter.lora_sites,
                config.locate_marker_id,
            )
            if self.adapter.inert_sites:
                logger.warning(
                    "%d of those %d sites sit behind a mixer this port has no"
                    " seam for, which reads its projections as bare tensors, so"
                    " no wrapped forward runs them and their delta is missing"
                    " from a Locate row. The seam bodies live in layers/.",
                    self.adapter.inert_sites,
                    self.adapter.lora_sites,
                )
        else:
            # Two different reasons land here, and the log has to tell them
            # apart: a fold this process switched off, and a backbone that
            # never shipped an adapter. The first one's tensors still load and
            # the audit still excuses them by suffix, which reads as a plain
            # backbone unless the line says otherwise.
            if config.adapter_rank:
                logger.info(
                    "the checkpoint carries a rank-%d Locate adapter, but"
                    " DECIDER_ADAPTER_FOLD is off, so no adapter sites are"
                    " built, the adapter's tensors load and go unread, and"
                    " every step runs the plain backbone",
                    config.adapter_rank,
                )
            else:
                logger.info(
                    "the adapted checkpoint carries no *.lora_a tensors, so no"
                    " adapter sites are built and every step runs the plain"
                    " backbone"
                )

    def head(self, x: Tensor) -> Tensor:
        """Returns the readout's scores of ``x``, accumulated in float32.

        Replaces both donor branches: the ``lm_head`` projection (absent, tied
        away) and the ``x @ embed_tokens.weight.T`` fallback, which would score
        the 248320-token vocabulary rather than the 255 answer codes. The cast
        is deliberate -- the reference runs the whole classifier in float32,
        and a bfloat16 matmul here quantizes logits whose top-two gaps the
        measurement is about.
        """
        logits = (
            x.cast(DType.float32) @ self.readout.weight.cast(DType.float32).T
        )
        return F.allgather(logits, tensor_axis=-1).local_shards[0]


class Qwen3_5DeciderDecoder(Module[[DecoderInputs[Tensor]], DecoderOutputs]):
    """The registered decider: text-only, one prefill, readout rows out.

    Takes the role contract's inputs and returns
    :class:`~max.pipelines.lib.interfaces.pipeline_model.ModuleV3Outputs` with
    ``next_token_logits`` of shape ``[batch, 255]``: the donor text model
    already gathers the logits at each row's last token, which is where
    protocol v4 puts the answer marker.
    """

    graph_name = "qwen3_5_decider"
    consumes_vision: ClassVar[bool] = False
    """No ``VISION`` module is registered, so the graph takes no image rows."""

    @classmethod
    def token_position_type(
        cls, config: object, device_ref: DeviceRef
    ) -> TensorType | None:
        """Returns the input type of the step's segment edges, or ``None``.

        Under :data:`~.layers.deltanet.STATE_RESTART_ENABLED` this decoder takes
        a static ``[blocks + 2]`` int32 tensor in the ``token_positions`` field,
        which :class:`~.batch_processor.Qwen3_5DeciderBatchProcessor` fills on the
        host each step. The segment edges have to reach the state kernels'
        ``input_row_offsets`` as a graph input: MAX faults in
        ``_Model_execute`` over a pair the graph reduces itself; the
        reproduction is recorded in :mod:`.batch_processor`. Off, no positions input is declared, so the graph's
        inputs and the batch processor's work stay the recorded ones.
        """
        if not STATE_RESTART_ENABLED:
            return None
        return TensorType(DType.int32, [RESTART_BLOCKS + 2], device=device_ref)

    checkpoints_recurrent_state: ClassVar[bool] = True
    """The gated-DeltaNet state leaves in the cache tree are what select the
    Jenga pool, as in the donor registration."""

    label_positions_declared = True
    """The graph takes the rows a label-scoring step gathers as an operand, so
    one shared row can answer every question marker in it.
    :class:`~.batch_processor.Qwen3_5DeciderBatchProcessor` fills it from
    :attr:`~max.pipelines.context.TextContext.label_positions`."""

    def __init__(self, config: Qwen3_5DeciderConfig) -> None:
        """Builds the model from the finalized config and audits the weights.

        Args:
            config: The port's config, which recorded the adapted checkpoint's
                tensor names when it finalized.
        """
        super().__init__()
        self.config = config
        self.kv_params = config.kv_params
        self.model = Qwen3_5DeciderTextModel(config)
        # Name-transparent, as the donor's registered decoder: the checkpoint's
        # weights are named by the text model itself, with no ``model.``
        # wrapper in front of them.
        self.model.name_transparent = True
        audit_weight_coverage(
            checkpoint_tensor_names=config.checkpoint_tensor_names,
            module=self,
        )

    def forward(self, inputs: DecoderInputs[Tensor]) -> DecoderOutputs:
        """Runs one prefill over the batch and returns the readout rows.

        The cache tree is the port's ``MultiKVCacheParams``: attention pages
        beside the gated-DeltaNet state. The pages are written and read inside
        this one call; the registration turns prefix caching and chunked
        prefill off, so a row is never split across steps and no step ever
        resumes a state it did not just compute.
        """
        mesh = self.model.mesh
        tokens = replicate(inputs.tokens, mesh)
        input_row_offsets = replicate(inputs.input_row_offsets, mesh)
        kv_tree = self.kv_params.unflatten_kv_inputs(
            iter(kv_input_values(tree.leaves(inputs.kv)))
        )
        assert isinstance(kv_tree, dict), (
            "the decider's cache tree is an attention leaf beside a"
            " recurrent-state leaf"
        )
        attn = kv_tree[ATTN_CACHE_KEY]
        state = distributed_state(
            tree.leaves(
                kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
            ),
            tokens.mapping,
        )
        kv_collection = PagedCacheValues.from_upstream(
            tree.leaves(attn, leaf=KVCacheInputsPerDevice), tokens.mapping
        )
        segment_labels: Tensor | None = None
        if SEGMENT_MASK_ENABLED or STATE_RESTART_ENABLED:
            # One label tensor for the step: the state block and its separator
            # are shared by every question chunk, and each chunk sees itself and
            # that state. The text model threads the tensor to the eight
            # full-attention layers, and the restarted DeltaNet reads the same
            # labels for its segment table.
            if self.model.restart is None:
                # No segment table to read, so the labels come off the tokens:
                # the state-once renderer ends every chunk with
                # ``<|fim_suffix|>``, and counting it finds the chunk ends.
                segment_labels = marker_segments(tokens, input_row_offsets)
            else:
                # The edges come in with the step, and the labels are read off
                # those edges: one boundary rule for the mask, the rewind and
                # the rope, so a step cannot put a chunk in one span and rewind
                # it in another. The offsets operand of the two state kernels is
                # the one value that cannot be a reduction over
                # ``Dim('total_seq_len')``. The table describes one row per step,
                # which is what the decisions route scores, and
                # ``Qwen3_5DeciderBatchProcessor.token_positions`` refuses a step that
                # batches more.
                if isinstance(input_row_offsets.shape[0], int):
                    if input_row_offsets.shape[0] != 2:
                        raise RuntimeError(
                            "DECIDER_STATE_RESTART traces one segment table per"
                            f" step and this step has"
                            f" {input_row_offsets.shape[0] - 1} rows; the"
                            " restarted path serves one request per forward"
                            " pass until the table carries a row axis"
                        )
                if inputs.token_positions is None:
                    raise RuntimeError(
                        "DECIDER_STATE_RESTART is set and this step has no"
                        " token_positions input, so the linear layers have no"
                        " segment edges to replay between."
                        " Qwen3_5DeciderDecoder.token_position_type() declares that"
                        " input and Qwen3_5DeciderBatchProcessor.token_positions()"
                        " fills it; a custom architecture that replaces either"
                        " leaves the rewound path with nothing to read"
                    )
                edges = replicate(inputs.token_positions, mesh)
                self.model.restart.value = edges
                self.model.restart.segment_index = segment_index_from_edges(
                    edges, tokens.shape[0], self.model.restart.blocks
                )
                segment_labels = attention_labels(
                    self.model.restart.segment_index
                )
                if POSITION_RESTART_ENABLED:
                    # The rope positions come off the same two operands the
                    # rewind assembles its spans with, so a step cannot put a
                    # chunk at one position and rewind it at another.
                    self.model.restart.positions = restarted_positions(
                        edges, self.model.restart.segment_index
                    )
        if self.model.adapter is not None:
            # One gate tensor for the step, read by all 248 delta sites and by
            # the marker swap: 1.0 in a row that holds the ``<|locate|>`` token,
            # 0.0 in every other row, so a decisions row and a Locate row can
            # share this graph and this step.
            self.model.adapter.gate.value = token_gate(
                tokens, input_row_offsets, self.model.adapter.marker_id
            )
        return self.model(
            tokens,
            input_row_offsets,
            inputs.return_n_logits,
            kv_collection,
            state,
            segment_ids=segment_labels if labels_reach_attention() else None,
            label_positions=inputs.label_positions,
        )
