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

"""Per-batch label scoring shared by the text generation pipelines.

A scoring request is an ordinary one-token generation whose context carries
``label_token_ids``. The pipelines call :meth:`BatchLabelScorer.score` on the
step's logits before the sampler runs and attach the result to
:attr:`TextGenerationOutput.label_log_probabilities`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.context import TextContext, TextGenerationContextType
from max.pipelines.modeling.types import RequestID
from max.pipelines.sampling import LabelScorer

from ..interfaces import ModelOutputs
from ..utils import CompilationTimer


def _rows_to_host(buffer: Buffer) -> npt.NDArray[np.float32]:
    """The step's gathered rows as float32 on the host.

    The channel leaves the graph in the model's dtype; bfloat16 has no
    numpy counterpart, so those rows widen through their bits.
    """
    host = buffer.to(CPU())
    if host.dtype == DType.bfloat16:
        bits = np.from_dlpack(host.view(DType.uint16))
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return host.to_numpy().astype(np.float32)


def _asked_labels(
    context: TextGenerationContextType,
) -> list[list[int] | None]:
    """Returns the candidate list of each logits row this context asks scored.

    A scoring context asks one row per entry of
    :attr:`TextContext.label_positions` when it names positions, and its
    next-token row otherwise. A context that is not scoring, or a scoring
    request whose prompt is still being chunked, asks nothing and gets one
    ``None``: only the step that completes the prompt holds the answer rows.

    Raises:
        ValueError: If the context names positions and candidate groups in
            different counts.
    """
    if (
        not isinstance(context, TextContext)
        or context.tokens.actively_chunked
        or context.tokens.generated_length != 0
    ):
        return [None]
    groups = context.label_token_id_groups
    positions = context.label_positions
    if positions is None:
        if groups is None:
            return (
                [list(context.label_token_ids)]
                if context.label_token_ids
                else [None]
            )
        raise ValueError(
            f"request {context.request_id} names {len(groups)} candidate "
            "groups and no label positions: a positions request names both"
        )
    if groups is None:
        if context.label_hidden:
            # The rows are reads, not scoring sites: the request wants the
            # hidden state of every position it named and scores no
            # candidates. The step still gathers those rows, and they come
            # back on ``label_hidden_rows``.
            return [None] * len(positions)
        raise ValueError(
            f"request {context.request_id} names {len(positions)} label "
            "positions and no candidate groups: every scored row needs its "
            "own candidate list"
        )
    if len(groups) != len(positions):
        raise ValueError(
            f"request {context.request_id} names {len(positions)} label "
            f"positions and {len(groups)} candidate groups"
        )
    return [list(group) for group in groups]


class BatchLabelScorer:
    """Scores the candidate labels of the scoring requests in a batch.

    The gather graph is compiled on the first batch that holds a scoring
    request, so servers that never see one pay nothing at startup.
    """

    def __init__(
        self,
        session: InferenceSession,
        device: Device,
        in_dtype: DType,
        vocab_size: int | None,
    ) -> None:
        """Creates a scorer; compilation is deferred to the first use.

        Args:
            session: The inference session that compiles the gather graph.
            device: The model device the logits live on.
            in_dtype: The dtype of the model's next-token logits.
            vocab_size: The tokenizer's token count, or ``None`` when the
                tokenizer cannot report it (scoring is then refused).
        """
        self._session = session
        self._device = device
        self._in_dtype = in_dtype
        self._vocab_size = vocab_size
        self._scorer: LabelScorer | None = None

    def score(
        self,
        model_outputs: ModelOutputs,
        flat_batch: Sequence[TextGenerationContextType],
    ) -> dict[RequestID, list[list[float]]]:
        """Scores the candidate labels of every scoring request in the batch.

        Must run before the sampler: its penalty and min-token processors edit
        the logits buffer in place.

        Args:
            model_outputs: Outputs of this step's forward pass. A batch whose
                contexts named positions scores the rows of
                :attr:`ModelOutputs.label_logits`; one that named none scores
                :attr:`ModelOutputs.next_token_logits`.
            flat_batch: The flattened context batch. A context contributes one
                logits row, or one per entry of
                :attr:`TextContext.label_positions`.

        Returns:
            Full-vocabulary label log-probabilities keyed by request id, one
            list per scored row in the order that request named them. Empty
            when the batch has none.

        Raises:
            ValueError: If the batch has a scoring request but the model
                returns variable logits or the vocabulary size is unknown, or a
                scoring request names a different number of positions and
                candidate groups.
        """
        # The readout gathers one logits row per named position, so the split
        # back to one entry per request lands here, where the counts were
        # named; a request contributes as many rows as it asked, with nothing
        # padded up to a batch maximum.
        asked = [_asked_labels(context) for context in flat_batch]
        row_labels = [
            labels for context_labels in asked for labels in context_labels
        ]
        if not any(row_labels):
            return {}
        if model_outputs.logit_offsets is not None:
            # Policy, not consequence: the decisions route refuses variable
            # logits, and a build that returns them would have to say which of
            # its rows are the scored ones, which the host already named.
            raise ValueError("label scoring does not support variable logits")
        # A model that gathers the rows the host named returns them on their own
        # channel, because ``next_token_logits`` stays one row per context for
        # the sampler. A step that named no positions has nothing on that
        # channel and scores the next-token rows, as it always did.
        logits = model_outputs.label_logits
        if logits is None:
            logits = model_outputs.next_token_logits
        if logits is None:
            logits = model_outputs.logits
        scores = self._label_scorer().score(logits, row_labels)
        results: dict[RequestID, list[list[float]]] = {}
        cursor = 0
        for context, context_labels in zip(flat_batch, asked, strict=True):
            rows = [
                row
                for row in scores[cursor : cursor + len(context_labels)]
                if row is not None
            ]
            cursor += len(context_labels)
            if rows:
                results[context.request_id] = rows
        return results

    def hidden_rows(
        self,
        model_outputs: ModelOutputs,
        flat_batch: Sequence[TextGenerationContextType],
    ) -> dict[RequestID, npt.NDArray[np.float32]]:
        """Returns the hidden rows the step gathered, per asking request.

        The same rows :meth:`score` reads candidates in, one stage before
        the head: a context that set :attr:`TextContext.label_hidden` reads
        them itself instead of naming candidates. Rows land in gather order,
        and each asking request gets its own slice. Gather order is the order
        :meth:`BatchProcessor.label_positions` names them: one entry per
        position the context named and one next-token row for every other
        context.

        Args:
            model_outputs: Outputs of this step's forward pass. A model
                without the ``label_hidden`` channel returns ``None`` there,
                and this answers nothing.
            flat_batch: The flattened context batch.

        Returns:
            Float32 host rows keyed by request id, only for asking
            contexts. Empty when the batch asks none or the model returns
            no channel.

        Raises:
            RuntimeError: If the gathered rows and the batch's named counts
                disagree, which means the model and the batch processor
                counted the step differently.
            ValueError: If a context names positions and candidate groups
                in different counts, or one without the other.
        """
        buffer = model_outputs.label_hidden
        if buffer is None:
            return {}
        asking = [
            context.request_id
            for context in flat_batch
            if isinstance(context, TextContext) and context.label_hidden
        ]
        if not asking:
            return {}
        rows = _rows_to_host(buffer)
        out: dict[RequestID, npt.NDArray[np.float32]] = {}
        cursor = 0
        for context in flat_batch:
            if not isinstance(context, TextContext):
                continue
            count = len(_asked_labels(context))
            if context.label_hidden:
                out[context.request_id] = rows[cursor : cursor + count]
            cursor += count
        if cursor != rows.shape[0]:
            raise RuntimeError(
                f"the step gathered {rows.shape[0]} hidden rows and the"
                f" batch names {cursor}: the model and the batch processor"
                " disagree on the gather"
            )
        return out

    def _label_scorer(self) -> LabelScorer:
        if self._scorer is None:
            if self._vocab_size is None:
                raise ValueError(
                    "label scoring needs the model's vocabulary size"
                )
            with CompilationTimer("label scorer") as timer:
                self._scorer = LabelScorer(
                    self._session,
                    self._device,
                    self._in_dtype,
                    self._vocab_size,
                )
                timer.mark_build_complete()
        return self._scorer
