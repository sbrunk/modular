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

"""Qwen3.5 batching that hands the rewound DeltaNet its segment edges.

:meth:`Qwen3_5DeciderBatchProcessor.token_positions` is the one override
:class:`Qwen3_5BatchProcessor` gains. MAX's batch interface fills that field on
the host each step, and a decoder that declares positions of another shape is
expected to override it
(``max/pipelines/lib/module_registration.py:399-413``,
``DecoderBatchProcessor.token_positions``). Under
:data:`~.layers.deltanet.STATE_RESTART_ENABLED` the declared shape is
``[blocks + 2]`` int32 and the values are the step's token edges, so the state
kernels' ``input_row_offsets`` operand arrives as a graph input.

Why the edges come from the host: a value built inside the graph from a
``Dim('total_seq_len')`` reduction and handed to ``gated_delta_conv1d_fwd`` or
``gated_delta_recurrence_fwd`` as ``input_row_offsets`` faults at
``fork/max/engine/api.py:145`` in ``_Model_execute``, before the graph's first
kernel launch reports anything.
A static-shape graph input answers them where a graph reduction cannot,
faulting the reduction over one layer and one 203-token row. The per-token labels stay in the graph:
:func:`~.layers.deltanet.segment_index_from_edges` turns this step's edges into
them, and :func:`~.layers.attention.marker_segments` is the fallback that counts
the marker token when the step has no edges. The attention mask and the
per-segment output assembly read whichever of the two ran.

Two rules can say where the spans are. A request that knows them states them
on its context's ``label_spans`` field: the token count of the state block and
of each question chunk, which :class:`SegmentBounds` reads back. With no such
statement, the edges are counted off the step's tokens, which matches the
labels :func:`~.layers.attention.marker_segments` derives token for token.
:meth:`Qwen3_5DeciderBatchProcessor.token_positions` logs which rule ran at
``debug``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.graph import TensorType
from max.pipelines.architectures.qwen3_5_modulev3.batch_processor import (
    Qwen3_5BatchProcessor,
)
from max.pipelines.context import TextContext

from .layers.attention import (
    CHUNK_MARKER_IDS,
    MARKER_TOKEN_ID,
    SEPARATOR_TOKEN_ID,
)
from .layers.deltanet import RESTART_BLOCKS, restart_edges

__all__ = [
    "Qwen3_5DeciderBatchProcessor",
    "SegmentBounds",
    "attach_segment_bounds",
    "segment_bounds",
    "segment_labels",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentBounds:
    """What a request states about its own spans, in tokens.

    ``spans`` is the token count of each span in row order: the state block with
    the ``<|fim_middle|>`` separator that closes it, then one count per question
    chunk with that chunk's marker included. Counts are row-local, so no token
    in the sequence has to mark a span end for the span to be found.
    """

    spans: tuple[int, ...]

    @property
    def total(self) -> int:
        """Tokens the spans cover, which is the step's token count."""
        return sum(self.spans)

    def edges(self, blocks: int) -> list[int]:
        """Returns the ``[blocks + 2]`` token edges of this row, padded.

        Cumulative span starts, padded to ``blocks + 2`` with the row's token
        count the way :func:`~.layers.deltanet.restart_edges` pads a row with
        fewer chunks than the compiled graph traces. A graph tracing ``blocks``
        rewinds cuts ``blocks + 1`` spans, and a request stating more spans has
        no edge the extra ones could arrive in.

        Args:
            blocks: How many rewinds the compiled graph traces, from
                :data:`~.layers.deltanet.RESTART_BLOCKS`.

        Raises:
            ValueError: When ``spans`` holds more than ``blocks + 1`` counts. A
                span past that bound shares the last edge with the span before
                it, and the questions inside both edges read each other. The
                message names the traced count and the requested count.
        """
        if len(self.spans) > blocks + 1:
            raise ValueError(
                f"the compiled graph traces {blocks} rewinds, which cuts a row"
                f" into {blocks + 1} spans, and this request states"
                f" {len(self.spans)} spans {self.spans}; the spans past"
                " that bound share one edge, so their questions read each"
                " other. Set DECIDER_RESTART_BLOCKS to"
                f" {len(self.spans) - 1} and rebuild, or send a request with"
                f" at most {blocks} questions"
            )
        edges: list[int] = [0]
        running = 0
        for span in self.spans[:blocks]:
            running += span
            edges.append(running)
        edges.append(self.total)
        while len(edges) < blocks + 2:
            edges.append(self.total)
        return edges


def attach_segment_bounds(context: Any, spans: Sequence[int]) -> SegmentBounds:
    """Puts the request's span counts on the context MAX built for it.

    The counts land on :attr:`TextContext.label_spans`, the field that
    carries them across the host-to-worker boundary with the rest of the
    context.
    """
    bounds = SegmentBounds(tuple(int(span) for span in spans))
    context.label_spans = list(bounds.spans)
    return bounds


def segment_bounds(context: Any) -> SegmentBounds | None:
    """Returns the span counts a request named on its context, or ``None``.

    A context that names none is a step that falls back to counting the
    marker token. Test fakes build contexts with ``__new__``, so the field is
    read with :func:`getattr`.
    """
    spans = getattr(context, "label_spans", None)
    return None if spans is None else SegmentBounds(tuple(spans))


def segment_labels(
    tokens: npt.NDArray[np.integer],
    marker_id: int | Sequence[int] = MARKER_TOKEN_ID,
    separator_id: int = SEPARATOR_TOKEN_ID,
) -> npt.NDArray[np.int32]:
    """Returns one row's segment labels, the host twin of ``marker_segments``.

    Tokens up to and including the ``<|fim_middle|>`` separator get -1, and
    chunk ``i`` (its question text plus the marker that ends it) gets
    ``i``. A token's chunk number is how many markers sit before it, so the
    row needs no second input to say where a chunk starts. ``marker_id``
    accepts either one id or the pair of chunk-ending ids; a row holds one
    dialect's markers, so counting the pair answers both.
    """
    ids = (marker_id,) if isinstance(marker_id, int) else tuple(marker_id)
    is_marker: npt.NDArray[np.bool_] = np.isin(tokens, np.asarray(ids))
    is_separator = tokens == separator_id
    markers_before: npt.NDArray[np.int64] = np.cumsum(is_marker) - is_marker
    separators_before = np.cumsum(is_separator) - is_separator
    return (markers_before - (separators_before == 0)).astype(np.int32)


class Qwen3_5DeciderBatchProcessor(Qwen3_5BatchProcessor):
    """Qwen3.5 batching with the step's segment edges in the positions field.

    Without :data:`~.layers.deltanet.STATE_RESTART_ENABLED` the graph declares
    no positions input, MAX never calls :meth:`token_positions`, and this class
    batches exactly like the donor it subclasses.

    Raises:
        ValueError: With the restart gate set and a step holding more than one
            row. :func:`~.layers.deltanet.restart_edges` counts one row's
            labels, and the rewound path serves one request per forward pass.
            Also raised when a request's stated spans do not add up to the
            tokens the step feeds, or state more spans than the compiled graph
            traces rewinds between. :meth:`SegmentBounds.edges` carries that
            second refusal, and :data:`~.layers.deltanet.RESTART_BLOCKS` is the
            count it compares against.
    """

    def token_positions(
        self, context_batch: Sequence[TextContext], position_type: TensorType
    ) -> npt.NDArray[np.integer[Any]]:
        """Returns the step's ``[blocks + 2]`` int32 token edges.

        The request's stated spans win when the context states them; the token
        count is the fallback. Either way the step logs which rule produced the
        table, because the labels, the mask, the rewind and the rope positions
        all follow it.
        """
        dtype = position_type.dtype.to_numpy()
        if not context_batch:
            # A warmup or an empty step: a table of empty spans, since the step
            # scores no tokens.
            return np.array(restart_edges([], RESTART_BLOCKS), dtype=dtype)
        if len(context_batch) != 1:
            raise ValueError(
                "DECIDER_STATE_RESTART traces one segment table per step and"
                f" this step batches {len(context_batch)} rows; the rewound"
                " DeltaNet serves one request per forward pass until the table"
                " names a row axis"
            )
        tokens = np.asarray(context_batch[0].tokens.active)
        bounds = segment_bounds(context_batch[0])
        if bounds is None:
            # Host-side ints for :func:`restart_edges`: the labels arrive as
            # the context's numpy array and the count is a plain walk.
            edges = restart_edges(
                [
                    int(label)
                    for label in segment_labels(tokens, CHUNK_MARKER_IDS)
                ],
                RESTART_BLOCKS,
            )
            logger.debug(
                "segment edges: counted any of tokens %s over %d tokens"
                " -> %s",
                CHUNK_MARKER_IDS,
                len(tokens),
                edges,
            )
        else:
            if bounds.total != len(tokens):
                raise ValueError(
                    f"the request states spans {bounds.spans} totaling"
                    f" {bounds.total} tokens and this step feeds"
                    f" {len(tokens)}; a span table has to cover the row it cuts"
                )
            edges = bounds.edges(RESTART_BLOCKS)
            logger.debug(
                "segment edges: the request's spans %s over %d tokens -> %s",
                bounds.spans,
                len(tokens),
                edges,
            )
        return np.array(edges, dtype=dtype)

    def label_positions(
        self,
        context_batch: Sequence[TextContext],
        input_row_offsets: npt.NDArray[np.integer[Any]],
    ) -> npt.NDArray[np.int64]:
        """Returns the rows a label-scoring step gathers, and logs the positions.

        A decisions request that shares one row names its markers on
        :attr:`~max.pipelines.context.TextContext.label_positions`. The graph
        ties those positions to the rows it gathered. A step
        whose contexts name nothing prints nothing and gathers each context's
        next-token row.
        """
        rows = super().label_positions(context_batch, input_row_offsets)
        named = [
            ctx.label_positions
            for ctx in context_batch
            if ctx.label_positions is not None
        ]
        if named:
            logger.debug(
                "label rows: prompt positions %s -> gathered rows %s over"
                " %d tokens",
                named[0],
                rows.tolist(),
                int(input_row_offsets[-1]),
            )
        return rows
