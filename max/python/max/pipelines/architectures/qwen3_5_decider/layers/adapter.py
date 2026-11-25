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

"""The Locate adapter as weight nodes plus the ops of ``LoRALinear.forward``.

``locate-span/adapter.safetensors`` holds 497 float32 tensors: ``lora_a``
``[rank, in_features]`` and ``lora_b`` ``[out_features, rank]`` for each of the
248 ``nn.Linear`` modules of ``backbone.language_model``, plus one
``embed_tokens.locate`` row of ``hidden_size``. The reference applies them only
while its adapter is ``enabled``: each ``LoRALinear.forward`` adds
``inputs.float() @ A.T @ B.T * (alpha / rank)`` to its projection's output and
casts that update back to the output's dtype, and ``MarkerEmbedding.forward``
answers the marker's id with the trained row. A decisions request runs with
them off, so its logits are those of the plain backbone.

The delta stays unmerged. ``W + BA`` folded into the bfloat16 base weight rounds
every delta entry to the bfloat16 grid, and the decisions task scores option
pairs whose gaps run near 1e-3, so the two terms are added at run time:
``base(x)`` in bfloat16, ``x_f32 @ A.T @ B.T * scale`` in float32, the sum cast
back once per projection.

The gate. A reference ``LoRALinear`` owns an ``enabled`` flag that
``LocateAdapter._enable`` sets per forward pass. A compiled graph has no such
flag: one trace runs every step, and a host-side flag cannot reach it.
``DecoderInputs`` brings tokens, row offsets, cache leaves, constants and
scratch, and the scratch buffers are workspace the graph's own kernels fill
(:mod:`max.pipelines.lib.decoder_scratch`), so nothing outside the graph writes
them per request. :func:`token_gate` therefore derives the gate from the tokens
the step already has: every token of a row that contains the ``<|locate|>``
marker gets 1.0, every token of a row without one gets 0.0. The shipped Locate
prompt appends that marker to each locate question
(``locate_config.json`` ``prompt``), and a protocol-v4 decisions row contains no
such token. One compiled graph serves both, in one batch if they arrive
together, and each row is gated on its own tokens. A gated-off row adds
``cast(0.0 * delta, dtype)``, which leaves every bit of the base result alone.

MAX's own LoRA serving route does not serve this adapter. It wraps a pipeline
model whose
module declares ``lora_targets``, and
``LoRAMetadata._validate_target_modules``
(``max/pipelines/lora/lora.py:342-373``) accepts ``q_proj``, ``k_proj``,
``v_proj`` and ``o_proj``; this adapter targets all 248 projections, and
``qwen3_5_decider`` declares no targets.

:func:`declare_lora` puts the two tensors on the module that already holds the
base weight, so a checkpoint name lands on the path the weight rules produce:
``language_model.layers.7.mlp.down_proj.lora_a`` becomes
``layers.7.mlp.down_proj.lora_a`` under the name-transparent text model.
:func:`install` puts those nodes on every projection of the built text model.

A site only produces a delta where something calls the projection's
``forward``. Three mixers reach their projections three ways, and each needs
its own seam: the MLP through :class:`~.mlp.LocateMLP`, which calls all three
projections where the donor fuses two of them into one matmul; the gated
DeltaNet, whose donor core reads its five projections as bare tensors, so
:class:`~.deltanet.LocateGatedDeltaNet` carries a body that adds them; and full
attention, whose core in
``max.pipelines.architectures.qwen3_5_modulev3.layers.attention`` reads four
projections as one ``stacked_matmul`` over concatenated weights and answers
three views of that one buffer, so
:func:`~.layers.attention._attention_core_with_deltas` adds each site's delta to
a destination of its own. :func:`install` re-classes the mixers that need one,
and counts every site it wrapped under the mixer family it sits under, so a
folded build logs ``mlp 96/96 self_attn 32/32 linear_attn 120/120``
(:meth:`AdapterSites.census`). A family that owns projections and has no live
site under it is a mixer with no seam body, and :func:`install` refuses
that build. The ``self_attn`` case is recorded in that function's docstring:
four projections read as one ``stacked_matmul``, where a delta written into a
``F.split`` view rewrites the rows beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.embedding import VocabParallelEmbedding
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.pipelines.architectures.qwen3_5_modulev3.layers.gated_deltanet import (
    GatedDeltaNet,
)
from max.tensor import Tensor

from .mlp import LocateMLP

__all__ = [
    "ADAPTER_SCALE",
    "AdapterGate",
    "AdapterSites",
    "LocateEmbedding",
    "declare_lora",
    "declare_marker",
    "delta",
    "install",
    "marker_rows",
    "pair_of",
    "token_gate",
    "with_delta",
]

ADAPTER_DELTA_DTYPE: DType = DType.float32
"""The dtype the delta tensors and their matmuls use.

``float32`` is what ``locate-span/adapter.safetensors`` stores every ``lora_a``
and ``lora_b`` as, and what ``LoRALinear.forward`` computes its update in:
``inputs.float()`` in, ``update.to(result.dtype)`` out. Declaring bfloat16
instead makes MAX auto-cast all 496 pair tensors on the way
in, with a ``load_state_dict auto-cast: 496 parameter(s) from DType.float32 to
DType.bfloat16 (precision loss)`` warning, and rounds every delta entry to the
base weight's grid.
"""

ADAPTER_ALPHA = 64
"""``lora.alpha`` of the shipped adapter, as ``locate-span/locate_config.json``
names it. No loader reads that field, so the value lives in this constant."""

ADAPTER_RANK = 64
"""``lora.rank`` of the shipped adapter, the other half of :data:`ADAPTER_SCALE`.

``qwen3_5_decider.model_config._read_adapter`` takes the rank a checkpoint
carries from its ``lora_a`` tensors. :func:`install` checks that rank against
this constant before sizing a pair: :data:`ADAPTER_SCALE` divides two
constants, and a checkpoint of another rank would score every delta at
``64 / rank`` of the trained value. The pair shapes, the parameter paths and the
marker row all validate at any rank, so the check is the only thing that stops
that build.
"""

ADAPTER_SCALE = ADAPTER_ALPHA / ADAPTER_RANK
"""The factor ``delta`` multiplies each update by: ``alpha / rank``, the scaling
``LoRALinear.forward`` applies. Both constants are 64 for the shipped adapter, so
the factor is 1.0.

A second adapter with a different ``lora.alpha`` needs this factor read from that
adapter's ``locate_config.json`` next to :data:`ADAPTER_RANK`; a pair sized at a
rank this constant has not priced is refused at :func:`install`.
"""


@dataclass
class AdapterGate:
    """The step's adapter gate, shared by every site of one compiled graph.

    One instance per compiled graph, bound once per step by
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderDecoder.forward`. It is a plain
    dataclass, not a tree node, so the tensor it names gets no parameter path
    and matches no checkpoint tensor; MAX's adapter layers hold their per-step
    routing the same way (``max/nn/common_layers/lora_wrapper.py:91-97``).
    """

    value: Tensor | None = None
    """``[total_seq_len]`` float32, one entry per token: 1.0 applies the delta
    to that token's row, 0.0 leaves the base projections. ``None`` before the
    decoder binds it, and :func:`with_delta` refuses that state."""


def _check_replicated(module: object) -> None:
    """Refuses a projection whose base weight sits on more than one device.

    Raises:
        NotImplementedError: If the base weight's mesh has more than one
            device. The delta is computed at full width; a build that splits
            ``out_features`` or ``in_features`` across devices has to split
            ``lora_b`` and ``lora_a`` the same way first.
    """
    weight = getattr(module, "weight", None)
    mesh = getattr(weight, "mesh", None)
    if mesh is not None and mesh.num_devices > 1:
        raise NotImplementedError(
            "the Locate delta is replicated across devices; a"
            " tensor-parallel build needs lora_a and lora_b sharded along the"
            " axis their base weight splits"
        )


def declare_lora(module: object, *, rank: int) -> None:
    """Puts this projection's ``lora_a`` and ``lora_b`` weight nodes on it.

    The widths come from the base weight the module already holds, so a
    declaration cannot disagree with the tensor the checkpoint names. Both
    additions are float32 while the tower runs bfloat16, as the checkpoint
    stores them and as the reference computes them: ``compile`` casts a
    state-dict entry to the dtype of the parameter that reads it
    (:func:`max.pipelines.lib.module_registration._cast_to_parameter_dtypes`),
    so a float32 declaration keeps the delta off the bfloat16 grid.

    Args:
        module: The projection module that holds the base weight, shaped
            ``[out_features, in_features]``.
        rank: The adapter rank, 64 for the shipped adapter.

    Raises:
        ValueError: If the base weight is not a two-axis matrix.
    """
    _check_replicated(module)
    weight = getattr(module, "weight", None)
    if weight is None or len(weight.shape) != 2:
        raise ValueError(
            f"{type(module).__name__} holds no [out_features, in_features]"
            " weight to size a delta against"
        )
    out_features, in_features = (int(dim) for dim in weight.shape)
    # The pair takes the base weight's placement, so the two additions stage on
    # the device that runs the projection. The decider serves one device, and
    # :func:`_check_replicated` refuses anything wider.
    placement = getattr(weight, "mapping", None) or weight.mesh
    dtype = ADAPTER_DELTA_DTYPE
    # Fresh attributes on a donor-built instance: setattr names that plainly,
    # and the noqa stops ruff (B010) from rewriting it into attribute access
    # the declared ``object`` type would not accept.
    delta_a = Tensor.zeros([rank, in_features], dtype=dtype).to(placement)
    delta_b = Tensor.zeros([out_features, rank], dtype=dtype).to(placement)
    setattr(module, "lora_a", delta_a)  # noqa: B010
    setattr(module, "lora_b", delta_b)  # noqa: B010


def declare_marker(module: object, *, hidden_size: int) -> None:
    """Puts the ``locate`` embedding row on the input embedding module."""
    row = Tensor.zeros([hidden_size], dtype=DType.float32)
    setattr(module, "locate", row)  # noqa: B010


def pair_of(module: object) -> tuple[Tensor, Tensor]:
    """Returns one projection's ``(lora_a, lora_b)`` pair.

    Raises:
        RuntimeError: If the module has no adapter tensors bound, which means
            a site was built without :func:`declare_lora`.
    """
    lora_a = getattr(module, "lora_a", None)
    lora_b = getattr(module, "lora_b", None)
    if lora_a is None or lora_b is None:
        raise RuntimeError(
            f"{type(module).__name__} has no lora_a/lora_b pair bound; the"
            " adapter sites are built by Qwen3_5DeciderTextModel"
        )
    return lora_a, lora_b


def delta(inputs: Tensor, pair: tuple[Tensor, Tensor], gate: Tensor) -> Tensor:
    """Returns this projection's adapter delta at ``inputs``, in their dtype.

    ``((x_f32 @ A.T) @ B.T) * alpha / rank`` in float32, as
    ``LoRALinear.forward`` computes it, scaled by the step's per-token gate and
    rounded to the input's dtype for the add. A gated-off token contributes 0.0.

    Args:
        inputs: What the base projection was called with, ``[rows, in]``.
        pair: The projection's ``(lora_a, lora_b)`` weight nodes.
        gate: The step's per-token gate, ``[rows]`` float32.
    """
    a, b = pair
    shrunk = inputs.cast(a.dtype) @ a.T
    return ((shrunk @ b.T) * ADAPTER_SCALE * gate.unsqueeze(-1)).cast(
        inputs.dtype
    )


def with_delta(
    base: Tensor,
    inputs: Tensor,
    pair: tuple[Tensor, Tensor],
    gate: Tensor | None,
) -> Tensor:
    """Returns one projection's output with the adapter delta added.

    With a gate of 0.0 the added term is 0.0 in the base's dtype, so ``base``
    comes back unchanged bit for bit.

    Args:
        base: The base projection's output, ``[rows, out_features]``.
        inputs: What that projection was called with, ``[rows, in_features]``.
        pair: The projection's ``(lora_a, lora_b)`` weight nodes.
        gate: The step's per-token gate, ``[rows]`` float32.

    Raises:
        RuntimeError: If the step bound no gate. Dropping the delta answers a
            Locate request with backbone rows.
    """
    if gate is None:
        raise RuntimeError(
            "no adapter gate is bound for this step; Qwen3_5DeciderDecoder.forward"
            " binds one with token_gate()"
        )
    return base + delta(inputs, pair, gate)


def marker_rows(
    tokens: Tensor,
    embeddings: Tensor,
    locate: Tensor,
    marker_id: int,
    gate: Tensor | None,
) -> Tensor:
    """Returns ``embeddings`` with the marker's rows taken from ``locate``.

    ``MarkerEmbedding.forward`` substitutes ``where(input == token_id, locate,
    values)`` while the adapter is enabled. The gate is ANDed in as a float
    multiply, so a row that reaches this embedding gated off keeps its base row
    even if it holds the marker id.

    Args:
        tokens: The ragged token ids, ``[rows]``.
        embeddings: The base embedding of ``tokens``, ``[rows, hidden]``.
        locate: The trained marker row, ``[hidden]`` float32.
        marker_id: The id whose rows read the trained vector, 248077 for the
            shipped adapter.
        gate: The step's per-token gate, ``[rows]`` float32.

    Raises:
        RuntimeError: If the step bound no gate.
    """
    if gate is None:
        raise RuntimeError(
            "no adapter gate is bound for this step; Qwen3_5DeciderDecoder.forward"
            " binds one with token_gate()"
        )
    hit = F.cast(F.equal(tokens, marker_id), DType.float32) * gate
    return F.where(
        (hit > 0.0).unsqueeze(-1),
        locate.cast(embeddings.dtype).to(embeddings.device),
        embeddings,
    )


def token_gate(tokens: Tensor, row_offsets: Tensor, marker_id: int) -> Tensor:
    """Returns the step's per-token adapter gate, ``[total_seq_len]`` float32.

    A token's gate is 1.0 when its own row contains the marker id and 0.0
    otherwise, so a batch that mixes a Locate row with a decisions row gates the
    two differently. The row index of each token comes from the
    histogram-cumsum trick MAX uses for request ids
    (``max/pipelines/architectures/deepseekV41_modulev3/layers/attention.py:231``):
    one ``scatter_add`` of row ends into a ``[total + 1]`` histogram and a
    ``cumsum``, with no ``[total, batch]`` comparison tile. A padded batch slot
    has no tokens, so it adds nothing to the marker counts.

    Args:
        tokens: The ragged token ids, ``[total_seq_len]``.
        row_offsets: Each row's start in ``tokens``, ``[batch_size + 1]``.
        marker_id: The id that turns the adapter on for its row.
    """
    device = row_offsets.mapping
    total = tokens.shape[0]
    off = F.cast(row_offsets, DType.int32)
    rows = off.shape[0] - 1
    ends = F.gather(
        off,
        F.range(0, rows, 1, out_dim=rows, dtype=DType.int32, device=device) + 1,
        axis=0,
    )
    positions = F.range(
        0,
        total + 1,
        1,
        out_dim=total + 1,
        dtype=DType.int32,
        device=device,
    )
    histogram = F.scatter_add(
        F.full([total + 1], 0, dtype=DType.int32, device=device),
        F.full([rows], 1, dtype=DType.int32, device=device),
        F.min(ends, positions[-1]),
        axis=0,
    )
    row_of_token = F.rebind(F.cumsum(histogram, axis=0)[:-1], [total])
    last_row = (
        F.range(
            0, rows + 1, 1, out_dim=rows + 1, dtype=DType.int32, device=device
        )[-1]
        - 1
    )
    row_of_token = F.min(row_of_token, F.reshape(last_row, [1]))

    markers = F.cast(F.equal(tokens, marker_id), DType.int32)
    counts = F.scatter_add(
        F.full([rows], 0, dtype=DType.int32, device=device),
        markers,
        row_of_token,
        axis=0,
    )
    per_row = F.cast(F.greater_equal(counts, 1), DType.float32)
    return F.rebind(F.gather(per_row, row_of_token, axis=0), [total])


class LocateEmbedding(VocabParallelEmbedding):
    """The input embedding, with the trained marker row behind the gate."""

    # Bound by install() when the fold is built, before any forward runs; the
    # declarations let this body and the weight audit name their types.
    locate: Tensor
    r"""The trained marker row, at the readout's ``<\|locate\|>`` id."""

    marker_id: int
    """The token id that turns the adapter on for its row."""

    gate: AdapterGate
    """The gate every site of this graph reads."""

    def forward(self, indices: Tensor) -> Tensor:
        """Returns the base rows, with a gated row's marker at ``locate``."""
        gate_value = self.gate.value
        if gate_value is None:
            # An unbound gate means the decoder never wired a folded step, so
            # no token selects the marker row: answer the base rows.
            return super().forward(indices)
        # The marker id sits past the base table: the release's tokenizer
        # registers its added tokens below it, and the adapter's trained row
        # is the only embedding that id has. The base gather therefore runs
        # over ids with the marker swapped for row 0, so no read leaves the
        # table; ``marker_rows`` then puts the trained vector in its place.
        hit = F.equal(indices, self.marker_id)
        safe = F.where(hit, F.zeros_like(indices), indices)
        return marker_rows(
            indices,
            super().forward(safe),
            self.locate,
            self.marker_id,
            gate_value,
        )


@dataclass
class AdapterSites:
    """What :func:`install` put on the text model, for the build log and audit.

    One plain dataclass held as one attribute of the text model: the module tree
    walk in ``Module.parameters`` stops at objects that are neither ``Tensor``
    nor ``Module``, so the gate tensor named inside gets no parameter path.
    """

    gate: AdapterGate = field(default_factory=AdapterGate)
    """The gate every site of this graph reads."""

    lora_sites: int = 0
    """Projections given a ``(lora_a, lora_b)`` pair."""

    inert_sites: int = 0
    """Sites whose wrapped ``forward`` no mixer calls.

    A site is inert when the mixer owning it runs its projections as bare
    tensors and has no seam body: a site under ``.linear_attn.`` where that
    mixer is a donor ``GatedDeltaNet``, under ``.self_attn.`` where it is a donor
    ``Qwen3_5Attention``, or under ``.mlp.`` where it is a donor
    ``QuantizedMLP``. Their ``(lora_a, lora_b)`` pair loads, sits in the audit's
    consumed set, and is read by no op. :func:`install` refuses a family whose
    sites are all inert, and leaves a partial count for
    :class:`qwen3_5_decider.decider.Qwen3_5DeciderDecoder` to warn about. A folded
    build on this tree reports zero.
    """

    sites_by_mixer: dict[str, int] = field(default_factory=dict)
    """Bound sites per mixer family, keyed by :data:`MIXER_FAMILIES` and
    :data:`OUTSIDE_MIXER`."""

    inert_by_mixer: dict[str, int] = field(default_factory=dict)
    """Inert sites per mixer family, the same keys as :attr:`sites_by_mixer`."""

    rank: int = 0
    """The rank those pairs were sized at."""

    marker_id: int = 0
    """The token id that gates the marker-embedding swap."""

    def census(self) -> str:
        """Returns each mixer family's live sites over its bound sites.

        :func:`install` refuses a family of this tree with a live count of zero,
        so a returned line names three families with a live site each.
        """
        parts = []
        for family in (*MIXER_FAMILIES, OUTSIDE_MIXER):
            bound = self.sites_by_mixer.get(family, 0)
            inert = self.inert_by_mixer.get(family, 0)
            parts.append(f"{family} {bound - inert}/{bound}")
        return " sites: " + " ".join(parts) + f"; total {self.lora_sites}"


MIXER_FAMILIES: tuple[str, ...] = ("mlp", "self_attn", "linear_attn")
"""The mixer each projection of ``backbone.language_model`` hangs off.

Each family reaches its projections a different way, so each has a seam class of
its own, and the site census counts by family. 96 ``mlp``, 32 ``self_attn`` and
120 ``linear_attn`` sites come out of the 32 blocks of this checkpoint's tower:
32 blocks of three MLP projections, 8 full-attention blocks of four, and 24
linear-attention blocks of five.
"""

OUTSIDE_MIXER = "outside"
"""The census key for a wrapped projection under none of :data:`MIXER_FAMILIES`."""


def _mixer_family(path: str) -> str:
    """Returns the mixer family a module path sits under, or :data:`OUTSIDE_MIXER`.

    ``path`` is what ``Module.descendants`` reports against the text model, for
    example ``layers.7.self_attn.q_proj``.
    """
    dotted = f".{path}."
    for family in MIXER_FAMILIES:
        if f".{family}." in dotted:
            return family
    return OUTSIDE_MIXER


_DELTA_CLASSES: dict[type, type] = {}
"""One delta subclass per projection class the tower uses, built once each."""


def _delta_class(cls: type[Module[Any, Any]]) -> type:
    """Returns a subclass of ``cls`` whose ``forward`` adds the site's delta.

    The tower's projections come from three classes in two packages
    (``max.nn.linear.Linear``, ``max.nn.common_layers.quant_linear.
    QuantizedLinear`` and the donor's ``qwen3_5_modulev3.layers.linear.
    Projection``). Re-classing an instance in place keeps the weight, bias, quant
    scales and sharding the donor allocated, and a generated subclass changes
    only ``forward``: the parameter paths the weight rules match are unchanged.
    """
    subclass = _DELTA_CLASSES.get(cls)
    if subclass is not None:
        return subclass

    def forward(self: Any, x: Tensor) -> Tensor:
        """Returns ``base(x) + gate * delta(x)``, the delta in float32."""
        return with_delta(
            cls.forward(self, x), x, pair_of(self), self.gate.value
        )

    subclass = type(
        f"Locate{cls.__name__}",
        (cls,),
        {
            "forward": forward,
            "__annotations__": {"lora_a": Tensor, "lora_b": Tensor},
        },
    )
    _DELTA_CLASSES[cls] = subclass
    return subclass


def install(
    text_model: Module[Any, Any],
    *,
    rank: int,
    marker_id: int,
    skip: tuple[str, ...] = (),
) -> AdapterSites:
    """Adds the adapter nodes to every projection of a built text model.

    A projection is any module of the built tree that holds a two-axis
    ``weight`` tensor, which is the reference's "every ``nn.Linear`` of
    ``language_model``" read off the model instead of a declared target list.
    RMSNorm and the DeltaNet scalars are skipped: their ``weight`` is one-axis.
    Each :class:`~max.nn.common_layers.quant_linear.QuantizedMLP` is re-classed
    to :class:`~.mlp.LocateMLP`, because the donor's body fuses ``gate_proj``
    and ``up_proj`` into one matmul and calls neither as a module, so their
    pairs would bind to a projection no op reads. Each linear-attention mixer
    is already a :class:`~.deltanet.LocateGatedDeltaNet`:
    :meth:`qwen3_5_decider.decider.Qwen3_5DeciderTextModel.__init__` builds it,
    and its core takes the five projections as ``ProjectionTensors`` and adds
    their deltas. A tree that still holds a donor
    :class:`~max.pipelines.architectures.qwen3_5_modulev3.layers.gated_deltanet.GatedDeltaNet`
    is refused. The four projections under ``self_attn`` get no re-class;
    they are live where ``Qwen3_5DeciderTextModel.__init__`` already built a
    :class:`~.layers.attention.DeciderAttention`, whose core adds the deltas
    itself.

    Args:
        text_model: The built text model whose projections take the delta.
        rank: The adapter rank, from the checkpoint's ``lora_a`` tensors.
        marker_id: The token id that gates the marker-embedding swap.
        skip: Attribute names of projections that take no delta. The decider's
            ``readout`` is a head the graph declares, not one of
            ``backbone.language_model``'s 248, so the adapter has no tensors
            for it.

    Raises:
        ValueError: If the text model has no ``embed_tokens`` module, or ``rank``
            is not :data:`ADAPTER_RANK`, whose quotient with
            :data:`ADAPTER_ALPHA` is the scale :func:`delta` applies.
        RuntimeError: If a mixer of this tree is a donor
            :class:`GatedDeltaNet`, or if a mixer family this tree holds has no
            live site under it, so every delta bound there is read by no op.
            The message carries :meth:`AdapterSites.census`.
    """
    if rank != ADAPTER_RANK:
        raise ValueError(
            f"adapter pairs would be sized at rank {rank} while"
            f" ADAPTER_SCALE is ADAPTER_ALPHA / ADAPTER_RANK ="
            f" {ADAPTER_ALPHA} / {ADAPTER_RANK}. A delta at that rank comes out"
            f" at {ADAPTER_ALPHA}/{rank} of what the shipped adapter's"
            " lora.alpha asks for, and the shapes, the parameter paths and the"
            " marker row all load. Set ADAPTER_ALPHA and ADAPTER_RANK to the"
            " two ``lora`` numbers of that adapter's"
            " locate-span/locate_config.json, or leave the fold off."
        )
    sites = AdapterSites(marker_id=marker_id, rank=rank)
    # Local imports: :mod:`.attention` and :mod:`.deltanet` read the delta ops
    # out of adapter.py, so neither import can sit at module scope.
    from .attention import DeciderAttention
    from .deltanet import LocateGatedDeltaNet

    descendants = list(text_model.descendants)
    # Qwen3_5DeciderTextModel.__init__ owns the linear-attention body: it builds
    # every linear_attention block as a LocateGatedDeltaNet, whatever
    # DECIDER_STATE_RESTART says, so install only checks the tree it was handed.
    # A donor GatedDeltaNet core reads its five projections as bare
    # ProjectionTensors, which leaves every pair bound under it unread.
    donor_deltanet = [
        path
        for path, module in descendants
        if isinstance(module, GatedDeltaNet)
        and not isinstance(module, LocateGatedDeltaNet)
    ]
    if donor_deltanet:
        raise RuntimeError(
            f"{len(donor_deltanet)} linear_attn mixers are still"
            f" {GatedDeltaNet.__name__}:"
            f" {', '.join(sorted(donor_deltanet)[:4])}."
            " Qwen3_5DeciderTextModel.__init__ builds each linear_attention"
            " block as a LocateGatedDeltaNet, the body that adds the five"
            " deltas; a donor core reads its projections as bare"
            " ProjectionTensors and no wrapped forward runs."
        )
    # The seam bodies go on before any site is counted, so one pass knows which
    # subtree is live.
    for _path, module in descendants:
        if isinstance(module, QuantizedMLP):
            module.__class__ = LocateMLP
    seam_mixers = {
        path
        for path, module in descendants
        if isinstance(
            module, (LocateMLP, DeciderAttention, LocateGatedDeltaNet)
        )
    }
    mixers = {family: 0 for family in MIXER_FAMILIES}
    for path, _ in descendants:
        family = path.rpartition(".")[2]
        if family in mixers:
            mixers[family] += 1
    for path, module in descendants:
        if path.rpartition(".")[2] in skip:
            continue
        if isinstance(module, Module) and (
            getattr(module, "q_proj", None)
            or getattr(module, "in_proj_qkv", None)
        ):
            # A mixer that runs its projections as bare tensors reads the gate
            # itself; LocateGatedDeltaNet and DeciderAttention are the two that
            # do. The bind is dynamic on purpose (a fresh attribute on a
            # donor-built instance), so it goes through setattr.
            setattr(module, "gate", sites.gate)  # noqa: B010
        weight = getattr(module, "weight", None)
        if (
            isinstance(module, Module)
            and isinstance(weight, Tensor)
            and len(weight.shape) == 2
        ):
            module.__class__ = _delta_class(type(module))
            declare_lora(module, rank=rank)
            setattr(module, "gate", sites.gate)  # noqa: B010
            sites.lora_sites += 1
            family = _mixer_family(path)
            sites.sites_by_mixer[family] = (
                sites.sites_by_mixer.get(family, 0) + 1
            )
            parent = path.rpartition(".")[0]
            if family != OUTSIDE_MIXER and parent not in seam_mixers:
                sites.inert_sites += 1
                sites.inert_by_mixer[family] = (
                    sites.inert_by_mixer.get(family, 0) + 1
                )
    seam_names = {
        "mlp": LocateMLP.__name__,
        "self_attn": DeciderAttention.__name__,
        "linear_attn": LocateGatedDeltaNet.__name__,
    }
    # A family the tree holds, with sites bound and every one of them inert,
    # loads the pairs, counts them consumed in the weight audit, and reads none
    # of them. Half of a family inert stays a count for the decider's warning
    # line.
    starved = [
        family
        for family in MIXER_FAMILIES
        if mixers[family]
        and sites.sites_by_mixer.get(family, 0)
        == sites.inert_by_mixer.get(family, 0)
    ]
    if starved:
        detail = "; ".join(
            f"{mixers[family]} {family} mixers hold"
            f" {sites.sites_by_mixer.get(family, 0)} adapter sites and every"
            f" one sits under a mixer that is not a {seam_names[family]}"
            for family in starved
        )
        raise RuntimeError(
            f"{detail}. No wrapped forward in that family runs, so the delta"
            " those pairs hold reaches no row, while"
            " ``Module.load_state_dict`` still counts each pair as consumed."
            f" install built the tree with{sites.census()}; the mixer swap is"
            " in Qwen3_5DeciderTextModel.__init__ and the seam bodies live in"
            " layers/."
        )
    embedding = getattr(text_model, "embed_tokens", None)
    if embedding is None:
        raise ValueError(
            "the text model has no embed_tokens module to take the"
            " language_model.embed_tokens.locate row"
        )
    embedding.__class__ = LocateEmbedding
    declare_marker(embedding, hidden_size=int(embedding.weight.shape[1]))
    embedding.gate = sites.gate
    embedding.marker_id = marker_id
    return sites
