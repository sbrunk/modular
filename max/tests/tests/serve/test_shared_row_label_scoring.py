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

"""``score_all`` scores one shared prompt once and hands each row its own row.

A request whose questions share one prompt names one marker per row, so the
route runs one ``pipeline.score`` per prompt and regroups the score rows by
position. One question naming one position is a positions request too: that case
reached ``score()`` with a flat candidate list and came back a 500. A row naming
two positions is refused, and a shared prompt is billed once.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decisions_prompt import QuestionView
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    EncodedRow,
    score_all,
    usage_totals,
)


class _StubPipeline:
    """Records what the route asked it to score, and answers with row scores."""

    def __init__(self) -> None:
        self.asked: list[tuple[Any, Any]] = []

    async def score(
        self,
        request: Any,
        label_token_ids: Any,
        label_positions: Any = None,
    ) -> LabelScoringOutput:
        self.asked.append((label_token_ids, label_positions))
        rows = [
            [float(10 * position + index) for index in range(len(candidates))]
            for position, candidates in zip(
                label_positions or [0],
                label_token_ids
                if label_positions is not None
                else [label_token_ids],
                strict=True,
            )
        ]
        return LabelScoringOutput(
            label_log_probabilities=rows[0] if len(rows) == 1 else [],
            prompt_token_count=100,
            cached_token_count=None,
            label_log_probability_rows=rows if len(rows) > 1 else None,
        )


class _Request:
    """The route state ``score_all`` reads off the HTTP request."""

    class _State:
        request_id = "r"
        request_timer = type("_Timer", (), {"start_ns": 0})()

    state = _State()
    url = type("_Url", (), {"path": "/v1/decisions"})()


def _encode(
    *rows: EncodedRow, question_ids: tuple[str, ...] = ()
) -> list[EncodedQuestion]:
    """One question per row, in the shape the prompt format returns."""
    ids = question_ids or tuple(f"q{index}" for index in range(len(rows)))
    return [
        EncodedQuestion(
            question_id=qid,
            view=QuestionView(
                kind="choice",
                question=None,
                names=[str(label) for label in row.label_ids],
                details=[None] * len(row.label_ids),
            ),
            rows=[row],
        )
        for qid, row in zip(ids, rows, strict=True)
    ]


async def _score_all(
    pipeline: Any, encoded: list[EncodedQuestion]
) -> list[list[LabelScoringOutput]]:
    """Runs ``score_all`` over the stub request state and stub pipeline."""
    return await score_all(
        cast(Any, _Request()), cast(Any, pipeline), "m", encoded
    )


def test_one_row_naming_one_position_is_scored() -> None:
    """A one-member position group keeps the positions shape, and answers."""
    pipeline = _StubPipeline()
    encoded = _encode(EncodedRow((1, 2, 3), [7, 9], [2]))
    results = asyncio.run(_score_all(pipeline, encoded))
    assert pipeline.asked == [([[7, 9]], [2])]
    assert results[0][0].label_log_probabilities == [20.0, 21.0]


def test_two_questions_on_one_prompt_are_scored_once() -> None:
    """Members of one prompt share a scoring request and get their own rows."""
    pipeline = _StubPipeline()
    shared = (1, 2, 3, 4)
    encoded = _encode(
        EncodedRow(shared, [7, 9], [1]),
        EncodedRow(shared, [3], [4]),
        question_ids=("first", "second"),
    )
    results = asyncio.run(_score_all(pipeline, encoded))
    assert pipeline.asked == [([[7, 9], [3]], [1, 4])]
    assert [row[0].label_log_probabilities for row in results] == [
        [10.0, 11.0],
        [40.0],
    ]


def test_a_row_naming_two_positions_is_a_request_error() -> None:
    """One question reads one row, and the second position would be dropped."""
    pipeline = _StubPipeline()
    encoded = _encode(EncodedRow((1, 2, 3, 4), [7, 9], [1, 4]))
    with pytest.raises(DecisionRequestError, match="names 2 label positions"):
        asyncio.run(_score_all(pipeline, encoded))
    assert pipeline.asked == []


def test_a_prompt_shared_by_two_questions_is_billed_once() -> None:
    """``usage_totals`` counts the prompt the scorer read once."""
    shared = (1, 2, 3, 4)
    encoded = _encode(
        EncodedRow(shared, [7, 9], [1]),
        EncodedRow(shared, [3], [4]),
        question_ids=("first", "second"),
    )
    results = [
        [
            LabelScoringOutput(
                label_log_probabilities=[0.0], prompt_token_count=4
            ),
            LabelScoringOutput(
                label_log_probabilities=[1.0], prompt_token_count=4
            ),
        ]
    ]
    assert usage_totals(results, encoded)[0] == 4
    assert usage_totals(results)[0] == 8
