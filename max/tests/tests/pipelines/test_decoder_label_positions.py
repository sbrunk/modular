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

"""One operand entry per logits row, so the counts agree at every step.

``DecoderBatchProcessor.label_positions`` fills the rows the readout gathers, and
``BatchLabelScorer.score`` pairs a candidate list against each of them. A context
whose prompt is still being chunked, or that has already generated, cannot
answer: it contributes one row and the scorer asks no candidates of it.
"""

from __future__ import annotations

import numpy as np
from label_context_fakes import FakeTokens, label_context
from max.pipelines.context import TextContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor
from max.pipelines.sampling import SamplingConfig, greedy_argmax_suffices


def _context(
    request_id: str,
    positions: list[int] | None = None,
    groups: list[list[int]] | None = None,
    tokens: FakeTokens | None = None,
) -> TextContext:
    """A context as :meth:`TokenGeneratorPipeline.score` leaves one.

    A positions request names every position's candidates in
    :attr:`~max.pipelines.context.TextContext.label_token_id_groups` and
    keeps the first group in
    :attr:`~max.pipelines.context.TextContext.label_token_ids`.
    """
    return label_context(
        request_id,
        label_token_ids=list(groups[0]) if groups else None,
        label_positions=positions,
        label_token_id_groups=groups,
        tokens=tokens,
    )


def _processor() -> DecoderBatchProcessor:
    """A batch processor, since the row rule reads no device state."""
    return DecoderBatchProcessor.__new__(DecoderBatchProcessor)


def test_an_unnamed_context_contributes_its_last_row() -> None:
    """The next-token row is the one the sampler already reads."""
    rows = _processor().label_positions(
        [_context("a"), _context("b", tokens=FakeTokens(active_length=3))],
        np.array([0, 5, 8], dtype=np.int64),
    )
    assert rows.tolist() == [4, 7]


def test_a_named_context_contributes_one_row_per_position() -> None:
    """A shared row naming three markers gathers three logits rows."""
    context = _context(
        "shared",
        positions=[2, 4, 9],
        groups=[[1], [2, 3], [4, 5, 6]],
        tokens=FakeTokens(active_length=10),
    )
    rows = _processor().label_positions([context], np.array([0, 10], np.int64))
    assert rows.tolist() == [2, 4, 9]


def test_a_chunked_context_naming_positions_contributes_one_row() -> None:
    """The step that cannot answer gathers one row and asks no candidates."""
    chunked = _context(
        "chunked",
        positions=[1, 4],
        groups=[[1], [2, 3]],
        tokens=FakeTokens(active_length=6, actively_chunked=True),
    )
    whole = _context(
        "whole",
        positions=[2, 5],
        groups=[[1], [2]],
        tokens=FakeTokens(active_length=8),
    )
    rows = _processor().label_positions(
        [chunked, whole], np.array([0, 6, 14], np.int64)
    )
    assert rows.tolist() == [5, 8, 11]
    assert len(rows) == 3


def test_a_generating_context_naming_positions_contributes_one_row() -> None:
    """A decode step of a scoring request gathers its next-token row alone."""
    generating = _context(
        "decoding",
        positions=[0, 3],
        groups=[[1], [2]],
        tokens=FakeTokens(active_length=1, generated_length=4),
    )
    rows = _processor().label_positions(
        [generating], np.array([9, 10], np.int64)
    )
    assert rows.tolist() == [9]


def test_a_positions_request_is_not_a_plain_greedy_step() -> None:
    """``greedy_argmax_suffices`` refuses what a positions request names.

    A positions request keeps a candidate group in ``label_token_ids``, the
    field that predicate reads, so the step keeps its sampler.
    """
    plain = _context("plain", tokens=FakeTokens(active_length=6))
    assert greedy_argmax_suffices(
        [plain], SamplingConfig(), constrained_decoding=False
    )
    shared = _context(
        "shared",
        positions=[2, 5],
        groups=[[7], [9, 11]],
        tokens=FakeTokens(active_length=6),
    )
    assert not greedy_argmax_suffices(
        [shared], SamplingConfig(), constrained_decoding=False
    )
