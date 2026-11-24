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

"""Locate rows: rendering, state-row naming, and the text-side answer.

The fake tokenizer from the joint v4 fakes answers offsets
piece-for-piece, so an assertion can name the exact token rows a state
occupies. No graph and no head run here: the span head's own cases cover
the scores, and the served span comparison is the GPU-side counterpart.
"""

from __future__ import annotations

import asyncio

import pytest
from max.pipelines.architectures.qwen3_5_decider.layers.attention import (
    SEPARATOR_TOKEN_ID,
)
from max.pipelines.architectures.qwen3_5_decider.span_head import LocateSettings
from max.serve.router._decisions_scoring import DecisionRequestError
from max.serve.router._joint_v4_format import (
    JointV4Format,
)
from max.serve.router._joint_v4_locate import assemble_spans, encode_locate
from tests.serve.joint_v4_fakes import (
    V4Tokenizer,
    _settings,
)

LOCATE_MARKER = "<|locate|>"
LOCATE_MARKER_ID = 248077

locate_settings = LocateSettings(
    width=128,
    proposals=128,
    max_span=128,
    threshold=0.5,
    marker=LOCATE_MARKER,
    marker_id=LOCATE_MARKER_ID,
    prompt="Question:\nLocate verbatim spans:\n{instructions}" + LOCATE_MARKER,
    prompt_format="joint-v4-locate",
)

STATE = "the payments team owns billing-eu"
INSTRUCTIONS = ["person names", "team names"]


def _encode(
    tokenizer: V4Tokenizer | None = None,
    instructions: list[str] | None = None,
    settings: LocateSettings = locate_settings,
    state: str = STATE,
):
    if tokenizer is None:
        tokenizer = V4Tokenizer()
    fmt = JointV4Format(_settings())
    separator = asyncio.run(fmt.separator_id(tokenizer))
    return asyncio.run(
        encode_locate(
            fmt.renderer(tokenizer),
            settings,
            tokenizer,
            state,
            INSTRUCTIONS if instructions is None else instructions,
            separator,
        )
    )


def test_one_row_ends_at_the_last_marker_and_names_the_others() -> None:
    row = _encode()
    assert row.prompt_ids[-1] == LOCATE_MARKER_ID
    assert row.marker_positions == [
        row.prompt_ids.index(LOCATE_MARKER_ID),
        len(row.prompt_ids) - 1,
    ]
    assert row.prompt_ids.count(SEPARATOR_TOKEN_ID) == 1  # separator once
    assert row.segments[0] == len(row.prompt_ids) - sum(row.segments[1:])
    assert sum(row.segments) == len(row.prompt_ids)
    assert row.prompt_ids[row.segments[0] - 1] == SEPARATOR_TOKEN_ID


def test_state_positions_cover_exactly_the_state_tokens() -> None:
    row = _encode()
    state_ids = V4Tokenizer()._ids(STATE)
    assert (
        row.prompt_ids[row.state_positions[0] : row.state_positions[-1] + 1]
        == state_ids
    )
    assert row.state_count == len(state_ids)
    # The fake is character level: one id and one span per non-space char.
    assert row.state_char_offsets[0][0] == 0
    assert row.state_char_offsets[-1][1] == len(STATE)


def test_a_special_token_in_the_state_cannot_open_a_marker_row() -> None:
    row = _encode(state="the <|locate|> token hides in the state")
    # Only the question chunks contribute real markers: two of them, and
    # the defused text inside the state contributes none.
    assert row.prompt_ids.count(LOCATE_MARKER_ID) == 2


def test_the_marker_rides_as_an_id_the_tokenizer_never_emits() -> None:
    # The fake's vocab does not know the literal at all, and the row still
    # carries the declared id exactly once per question: the encoder appends
    # the id, it does not encode the text.
    row = _encode(tokenizer=V4Tokenizer())
    assert row.prompt_ids.count(LOCATE_MARKER_ID) == len(INSTRUCTIONS)
    assert row.prompt_ids[-1] == LOCATE_MARKER_ID


def test_an_oversized_row_is_refused() -> None:
    tokenizer = V4Tokenizer(max_length=10)
    with pytest.raises(DecisionRequestError, match="context length"):
        _encode(tokenizer=tokenizer)


def test_spans_trim_merge_and_sort_by_position() -> None:
    state = "alpha beta  gamma"
    offsets = [(0, 5), (6, 10), (10, 12), (12, 17)]
    # tokens: alpha(0-5) beta(6-10) the pair of spaces (10-12) gamma(12-17)
    spans = assemble_spans(
        [(0, 1, 0.7), (0, 2, 0.4), (0, 2, 0.9), (2, 2, 0.8), (3, 3, 0.6)],
        offsets,
        state,
    )
    # The whitespace-only candidate (2,2) trims away; the three spans that
    # end at beta merge to (0,10) at their largest probability.
    assert [(a, b, text) for a, b, text, _ in spans] == [
        (0, 10, "alpha beta"),
        (12, 17, "gamma"),
    ]
    assert [p for _, _, _, p in spans] == [0.9, 0.6]
