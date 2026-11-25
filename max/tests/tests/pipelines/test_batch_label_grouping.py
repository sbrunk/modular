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

"""The batch scorer returns one score list per row a request asked.

``BatchLabelScorer.score`` pairs logits rows to candidate lists, and a shared
decisions row asks one row per named marker. The regroup by request has to agree
with the rows the graph gathered, and a request that asked one position still
gets its scores in the single-row shape the existing callers read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces import ModelOutputs
from max.pipelines.lib.pipeline_variants._label_scoring import (
    BatchLabelScorer,
    _asked_labels,
)


@dataclass
class _Tokens:
    """The two token counters the scorer reads."""

    actively_chunked: bool = False
    generated_length: int = 0


@dataclass
class _Outputs:
    """The output fields the scorer reads; the logits buffer is a stand-in."""

    next_token_logits: Any = "next-token rows"
    label_logits: Any = None
    label_hidden: Any = None
    logit_offsets: Any = None


def _outputs(**fields: Any) -> ModelOutputs:
    """The output fields the scorer reads, as the stand-in buffer."""
    return cast(ModelOutputs, _Outputs(**fields))


class _StubLabelScorer:
    """Records the logits buffer and the per-row candidates, one row per entry."""

    def __init__(self) -> None:
        self.asked: list[list[int] | None] = []
        self.chosen: Any = None

    def score(
        self, logits: Any, label_token_ids: list[list[int] | None]
    ) -> list[list[float] | None]:
        self.chosen = logits
        self.asked = [list(ids) if ids else None for ids in label_token_ids]
        return [
            [float(row), *[float(len(ids))]] if ids else None
            for row, ids in enumerate(label_token_ids)
        ]


def _scorer() -> tuple[BatchLabelScorer, _StubLabelScorer]:
    """A batch scorer whose gather graph is replaced by the stub."""
    batch_scorer = BatchLabelScorer.__new__(BatchLabelScorer)
    stub = _StubLabelScorer()
    batch_scorer._scorer = stub  # type: ignore[assignment]
    return batch_scorer, stub


def _context(
    request_id: str,
    label_token_ids: list[int] | None = None,
    label_positions: list[int] | None = None,
    label_token_id_groups: list[list[int]] | None = None,
    label_hidden: bool = False,
) -> TextContext:
    """A scoring context built without a tokenizer or a token buffer."""
    context = TextContext.__new__(TextContext)
    context.request_id = request_id  # type: ignore[assignment]
    context.tokens = _Tokens()  # type: ignore[assignment]
    context.label_token_ids = label_token_ids
    context.label_positions = label_positions
    context.label_token_id_groups = label_token_id_groups
    context.label_hidden = label_hidden
    return context


def test_one_position_keeps_the_single_score_list() -> None:
    """A request that named no position contributes one row of scores."""
    batch_scorer, stub = _scorer()
    results = batch_scorer.score(_outputs(), [_context("a", [7, 9])])
    assert results == {"a": [[0.0, 2.0]]}
    assert stub.asked == [[7, 9]]


def test_named_positions_return_one_list_per_position() -> None:
    """The shared row asks three markers and gets three score rows back."""
    batch_scorer, stub = _scorer()
    context = _context(
        "shared",
        label_token_ids=[11, 12],
        label_positions=[4, 9, 13],
        label_token_id_groups=[[11, 12], [3], [5, 6, 8]],
    )
    results = batch_scorer.score(_outputs(), [context])
    assert results == {"shared": [[0.0, 2.0], [1.0, 1.0], [2.0, 3.0]]}
    assert stub.asked == [[11, 12], [3], [5, 6, 8]]


def test_a_generating_context_in_the_batch_asks_nothing() -> None:
    """The generating row still occupies one logits row and gets no scores."""
    batch_scorer, stub = _scorer()
    generating = _context("gen")
    scoring = _context(
        "shared",
        label_token_ids=[1],
        label_positions=[2, 5],
        label_token_id_groups=[[1], [2, 3]],
    )
    results = batch_scorer.score(_outputs(), [generating, scoring])
    assert results == {"shared": [[1.0, 1.0], [2.0, 2.0]]}
    assert stub.asked == [None, [1], [2, 3]]


def test_positions_and_candidate_groups_have_to_agree() -> None:
    """A count mismatch is refused before the gather graph runs."""
    batch_scorer, stub = _scorer()
    context = _context(
        "broken",
        label_token_ids=[1],
        label_positions=[2, 5],
        label_token_id_groups=[[1]],
    )
    with pytest.raises(ValueError, match="2 label positions and 1 candidate"):
        batch_scorer.score(_outputs(), [context])
    assert stub.asked == []


def test_positions_without_candidate_groups_are_refused() -> None:
    """Half a positions request is refused before any launch."""
    batch_scorer, stub = _scorer()
    context = _context("half", label_positions=[2, 5])
    with pytest.raises(ValueError, match="no candidate groups"):
        batch_scorer.score(_outputs(), [context])
    assert stub.asked == []


def test_candidate_groups_without_positions_are_refused() -> None:
    """Candidate groups with no positions are refused before any launch."""
    batch_scorer, stub = _scorer()
    context = _context("no-positions", label_token_id_groups=[[1], [2, 3]])
    with pytest.raises(ValueError, match="2 candidate groups and no label"):
        batch_scorer.score(_outputs(), [context])
    assert stub.asked == []


def test_the_disagreement_names_the_request_and_both_counts() -> None:
    """The message reads as this request's, and states both counts."""
    context = _context(
        "half-asked",
        label_token_ids=[1],
        label_positions=[2, 5],
        label_token_id_groups=[[1]],
    )
    with pytest.raises(
        ValueError,
        match="request half-asked names 2 label positions and 1 candidate "
        "groups",
    ):
        _asked_labels(context)


def test_named_rows_come_from_the_label_channel() -> None:
    """A step that named positions scores the label channel."""
    batch_scorer, stub = _scorer()
    context = _context(
        "shared",
        label_token_ids=[1],
        label_positions=[2, 5],
        label_token_id_groups=[[1], [2, 3]],
    )
    results = batch_scorer.score(_outputs(label_logits="named rows"), [context])
    assert results == {"shared": [[0.0, 1.0], [1.0, 2.0]]}
    assert stub.chosen == "named rows"


def test_no_label_channel_falls_back_to_the_next_token_rows() -> None:
    """A build with no label channel scores what it always scored."""
    batch_scorer, stub = _scorer()
    results = batch_scorer.score(
        _outputs(next_token_logits="next-token rows", label_logits=None),
        [_context("a", [7, 9])],
    )
    assert results == {"a": [[0.0, 2.0]]}
    assert stub.chosen == "next-token rows"


def test_variable_logits_are_still_refused() -> None:
    """Rows the host did not choose stay out: the offsets come from the model."""
    batch_scorer, _ = _scorer()
    with pytest.raises(ValueError, match="does not support variable logits"):
        batch_scorer.score(
            _outputs(logit_offsets=object()), [_context("a", [7])]
        )


def test_a_row_read_names_no_candidates() -> None:
    """A request asking for hidden rows scores nothing and reads them itself."""
    batch_scorer, stub = _scorer()
    context = _context("rows", label_positions=[2, 5, 9], label_hidden=True)
    assert _asked_labels(context) == [None, None, None]
    results = batch_scorer.score(_outputs(label_logits="named rows"), [context])
    assert results == {}
    assert stub.asked == []


def test_positions_without_candidates_are_still_refused() -> None:
    """The exemption belongs to row reads, not to scoring requests."""
    context = _context("a", label_positions=[2, 5])
    with pytest.raises(ValueError, match="no candidate groups"):
        _asked_labels(context)


def test_hidden_rows_are_sliced_by_the_named_counts() -> None:
    """One gather block, split back per request in gather order."""
    import numpy as np
    from max.driver import Buffer

    batch_scorer, _ = _scorer()
    rows = np.arange(12.0, dtype=np.float32).reshape(6, 2)
    buffer = Buffer.from_numpy(np.ascontiguousarray(rows))
    batch = [
        _context("a", label_positions=[0, 1, 2], label_hidden=True),
        _context("b"),
        _context("c", label_positions=[4, 5]),
    ]
    out = batch_scorer.hidden_rows(_outputs(label_hidden=buffer), batch)
    assert len(out) == 1
    np.testing.assert_array_equal(list(out.values())[0], rows[:3])


def test_a_gather_the_batch_cannot_account_for_raises() -> None:
    """The model and the batch processor must count the same rows."""
    import numpy as np
    from max.driver import Buffer

    batch_scorer, _ = _scorer()
    buffer = Buffer.from_numpy(np.zeros([5, 2], dtype=np.float32))
    batch = [_context("a", label_positions=[0, 1, 2], label_hidden=True)]
    with pytest.raises(RuntimeError, match="disagree on the gather"):
        batch_scorer.hidden_rows(_outputs(label_hidden=buffer), batch)
