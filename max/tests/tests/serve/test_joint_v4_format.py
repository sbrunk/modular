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

"""Joint protocol v4 format: settings, rendering, and the readout read.

Fixtures live in ``tests/serve/joint_v4_fakes.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from max.pipelines.architectures.qwen3_5_decider.layers.attention import (
    MARKER_TOKEN_ID,
    SEPARATOR_TOKEN_ID,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    EncodedRow,
)
from max.serve.router._joint_v4_format import (
    DECISION_CONFIG_FILE,
    ISOLATED_CHUNKS,
    JOINT_CONFIG_FILE,
    MARKER,
    SEPARATOR,
    JointV4Format,
    find_joint_v4_settings,
    option_rows,
    parse_joint_v4_settings,
)
from tests.serve.joint_v4_fakes import (
    CODE_IDS,
    CODES,
    V4Tokenizer,
    _choice,
    _decision,
    _joint,
    _score,
    _settings,
    _yes_no,
)

# --------------------------------------------------------------------- #
# The two config files
# --------------------------------------------------------------------- #


def test_settings_carry_the_protocol_readout() -> None:
    settings = parse_joint_v4_settings(_joint(), _decision())
    assert settings.version == 4
    assert settings.codes == CODES
    assert settings.code_token_ids == CODE_IDS
    assert settings.temperature == 1.25
    assert settings.max_questions is None


@pytest.mark.parametrize(
    ("joint", "decision", "message"),
    [
        (_joint(template="other-v1"), _decision(), r"is not .* only renderer"),
        (_joint(version=5), _decision(), r"is not served"),
        (_joint(marker="<|weird|>"), _decision(), r"is not protocol v4's"),
        (
            _joint(),
            _decision(token_ids=[1, 2]),
            r"4 codes for 2 token ids",
        ),
        (
            _joint(),
            _decision(
                codes=[f"c{i}" for i in range(256)],
                token_ids=list(range(256)),
            ),
            r"exceeds the 255-row readout",
        ),
        (_joint(), _decision(temperature=0.0), r"not positive"),
    ],
)
def test_parse_refuses_a_checkpoint_it_cannot_render(
    joint: dict[str, Any], decision: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_joint_v4_settings(joint, decision)


def test_find_reads_a_checkpoint_dir_and_ignores_the_rest(
    tmp_path: Path,
) -> None:
    assert find_joint_v4_settings(str(tmp_path)) is None
    (tmp_path / JOINT_CONFIG_FILE).write_text(json.dumps(_joint()))
    with pytest.raises(ValueError, match=DECISION_CONFIG_FILE):
        find_joint_v4_settings(str(tmp_path))
    (tmp_path / DECISION_CONFIG_FILE).write_text(json.dumps(_decision()))
    settings = find_joint_v4_settings(str(tmp_path))
    assert settings is not None and settings.codes == CODES


# --------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_one_question_gets_its_own_row_ended_by_the_marker() -> None:
    format_ = JointV4Format(_settings())
    encoded = await format_.encode(
        V4Tokenizer(), "the state", [("q", _choice())], {}
    )
    assert len(encoded) == 1 and len(encoded[0].rows) == 1
    row = encoded[0].rows[0]
    assert row.label_positions is None
    assert row.prompt_ids[-1] == MARKER_TOKEN_ID
    assert row.prompt_ids.count(MARKER_TOKEN_ID) == 1
    assert row.prompt_ids.count(SEPARATOR_TOKEN_ID) == 1
    # A choice's candidates are readout rows in option order.
    assert row.label_ids == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_yes_no_question_reads_its_rows_swapped() -> None:
    format_ = JointV4Format(_settings())
    encoded = await format_.encode(
        V4Tokenizer(), "the state", [("q", _yes_no())], {}
    )
    assert encoded[0].rows[0].label_ids == [1, 0]


def test_option_rows_puts_the_no_line_first() -> None:
    lines, rows = option_rows(_yes_no())
    assert lines == ["No / false", "Yes / true"]
    assert rows == [1, 0]
    lines, rows = option_rows(_score("low", "mid", "high"))
    assert lines == ["low", "mid", "high"] and rows == [0, 1, 2]


@pytest.mark.asyncio
async def test_two_questions_share_one_row_and_name_their_markers() -> None:
    format_ = JointV4Format(_settings())
    encoded = await format_.encode(
        V4Tokenizer(),
        "the state once",
        [("a", _choice("x", "y", "z")), ("b", _choice("p", "q"))],
        {},
    )
    assert [item.rows[0].prompt_ids for item in encoded] == [
        encoded[0].rows[0].prompt_ids
    ] * 2
    row_a, row_b = (item.rows[0] for item in encoded)
    prompt = row_a.prompt_ids
    markers = [
        index for index, token in enumerate(prompt) if token == MARKER_TOKEN_ID
    ]
    assert len(markers) == 2
    assert row_a.label_positions == [markers[0]]
    assert row_b.label_positions == [markers[1]]
    # The state and its separator appear once for the whole request.
    assert prompt.count(SEPARATOR_TOKEN_ID) == 1
    assert row_a.label_ids == [0, 1, 2] and row_b.label_ids == [0, 1]


@pytest.mark.asyncio
async def test_more_questions_than_one_row_isolates_get_their_own_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    format_ = JointV4Format(_settings())
    items = [
        (f"q{index}", _choice("x", "y")) for index in range(ISOLATED_CHUNKS + 1)
    ]
    encoded = await format_.encode(V4Tokenizer(), "state", items, {})
    for item in encoded:
        row = item.rows[0]
        assert row.label_positions is None
        # Each question carries its own context: one separator, one marker.
        assert row.prompt_ids.count(SEPARATOR_TOKEN_ID) == 1
        assert row.prompt_ids.count(MARKER_TOKEN_ID) == 1
        assert row.prompt_ids[-1] == MARKER_TOKEN_ID


@pytest.mark.asyncio
async def test_special_tokens_in_the_state_are_defused() -> None:
    format_ = JointV4Format(_settings())
    encoded = await format_.encode(
        V4Tokenizer(),
        f"a state carrying {MARKER} and {SEPARATOR} inline",
        [("q", _choice("x", "y"))],
        {},
    )
    prompt = encoded[0].rows[0].prompt_ids
    assert prompt.count(SEPARATOR_TOKEN_ID) == 1  # the protocol's own only
    assert prompt.count(MARKER_TOKEN_ID) == 1


@pytest.mark.asyncio
async def test_a_template_that_moves_the_state_is_refused() -> None:
    format_ = JointV4Format(_settings())
    with pytest.raises(
        DecisionRequestError, match="the state cannot be rendered"
    ):
        await format_.encode(
            V4Tokenizer(shift_state=True),
            "state",
            [("q", _choice("x", "y"))],
            {},
        )


@pytest.mark.asyncio
async def test_a_row_over_the_context_length_names_its_question() -> None:
    format_ = JointV4Format(_settings())
    with pytest.raises(DecisionRequestError, match=r"question '.?':.*context"):
        await format_.encode(
            V4Tokenizer(max_length=24),
            "a state with quite a few words in it",
            [("a", _choice("x", "y")), ("b", _choice("x", "y", "z", "c"))],
            {},
        )


@pytest.mark.asyncio
async def test_a_question_wider_than_the_readout_is_refused() -> None:
    format_ = JointV4Format(_settings())
    assert format_.limits.choice_options == len(CODES)
    with pytest.raises(DecisionRequestError, match="1 to 4 are supported"):
        await format_.encode(
            V4Tokenizer(),
            "state",
            [("q", _choice("a", "b", "c", "d", "e"))],
            {},
        )


@pytest.mark.asyncio
async def test_a_code_that_is_not_its_token_refuses_the_server() -> None:
    format_ = JointV4Format(_settings())
    tokenizer = V4Tokenizer(
        code_ids={**dict(zip(CODES, CODE_IDS, strict=True)), "c2": 999}
    )
    with pytest.raises(DecisionRequestError, match="answer code 'c2'"):
        await format_.encode(
            tokenizer,
            "state",
            [("q", _choice("x", "y"))],
            {},
        )


@pytest.mark.asyncio
async def test_a_marker_numbered_elsewhere_is_refused() -> None:
    format_ = JointV4Format(_settings())
    with pytest.raises(DecisionRequestError, match="segment machinery"):
        await format_.encode(
            V4Tokenizer(marker_id=1234),
            "state",
            [("q", _choice("x", "y"))],
            {},
        )


def test_declared_max_questions_bounds_the_request() -> None:
    settings = _settings(joint={"max_questions": 2})
    assert JointV4Format(settings).limits.questions == 2


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #


def test_scores_apply_the_checkpoint_temperature_once() -> None:
    format_ = JointV4Format(_settings())
    view = _choice("x", "y", "z")
    question = EncodedQuestion("q", view, [EncodedRow((1, 2), [0, 1, 2])])
    log_probabilities = [-1.0, -2.0, -3.0]
    scored = format_.scores(
        question,
        [
            LabelScoringOutput(
                label_log_probabilities=log_probabilities,
                prompt_token_count=2,
            )
        ],
        None,
    )
    temperature = 1.25
    scaled = [value / temperature for value in log_probabilities]
    weights = [math.exp(value) for value in scaled]
    total = sum(weights)
    assert [round(value, 6) for value in scored.probabilities] == [
        round(value / total, 6) for value in weights
    ]
