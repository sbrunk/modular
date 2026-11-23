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

from dataclasses import dataclass

import numpy as np
from max.pipelines.context import TextContext
from max.pipelines.lib.module_registration import DecoderBatchProcessor


@dataclass
class _Tokens:
    """The three token counters the row rule reads."""

    active_length: int = 5
    actively_chunked: bool = False
    generated_length: int = 0


def _context(
    request_id: str,
    positions: list[int] | None = None,
    groups: list[list[int]] | None = None,
    tokens: _Tokens | None = None,
) -> TextContext:
    """A context built without a tokenizer, holding the scoring fields."""
    context = TextContext.__new__(TextContext)
    context.request_id = request_id  # type: ignore[assignment]
    context.tokens = tokens or _Tokens()  # type: ignore[assignment]
    context.label_token_ids = list(groups[0]) if groups else None
    context.label_positions = positions
    context.label_token_id_groups = groups
    return context


def _processor() -> DecoderBatchProcessor:
    """A batch processor, since the row rule reads no device state."""
    return DecoderBatchProcessor.__new__(DecoderBatchProcessor)


def test_an_unnamed_context_contributes_its_last_row() -> None:
    """The next-token row is the one the sampler already reads."""
    rows = _processor().label_positions(
        [_context("a"), _context("b", tokens=_Tokens(active_length=3))],
        np.array([0, 5, 8], dtype=np.int64),
    )
    print(f"unnamed contexts: rows {rows.tolist()}")
    assert rows.tolist() == [4, 7]


def test_a_named_context_contributes_one_row_per_position() -> None:
    """A shared row naming three markers gathers three logits rows."""
    context = _context(
        "shared",
        positions=[2, 4, 9],
        groups=[[1], [2, 3], [4, 5, 6]],
        tokens=_Tokens(active_length=10),
    )
    rows = _processor().label_positions([context], np.array([0, 10], np.int64))
    print(f"named context: rows {rows.tolist()}")
    assert rows.tolist() == [2, 4, 9]


def test_a_chunked_context_naming_positions_contributes_one_row() -> None:
    """The step that cannot answer gathers one row and asks no candidates."""
    chunked = _context(
        "chunked",
        positions=[1, 4],
        groups=[[1], [2, 3]],
        tokens=_Tokens(active_length=6, actively_chunked=True),
    )
    whole = _context(
        "whole",
        positions=[2, 5],
        groups=[[1], [2]],
        tokens=_Tokens(active_length=8),
    )
    rows = _processor().label_positions(
        [chunked, whole], np.array([0, 6, 14], np.int64)
    )
    print(f"chunked plus whole: rows {rows.tolist()}")
    assert rows.tolist() == [5, 8, 11]
    assert len(rows) == 3


def test_a_generating_context_naming_positions_contributes_one_row() -> None:
    """A decode step of a scoring request gathers its next-token row alone."""
    generating = _context(
        "decoding",
        positions=[0, 3],
        groups=[[1], [2]],
        tokens=_Tokens(active_length=1, generated_length=4),
    )
    rows = _processor().label_positions(
        [generating], np.array([9, 10], np.int64)
    )
    print(f"generating context: rows {rows.tolist()}")
    assert rows.tolist() == [9]
