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

"""Ceiling tests: the prompt format owns how many candidates fit a question."""

from __future__ import annotations

import pytest
from max.serve.router._chat_format import ChatFormat
from max.serve.router._decider_format import DeciderFormat
from max.serve.router._decider_prompt import (
    MAX_DECIDER_OPTIONS,
    DeciderSettings,
)
from max.serve.router._decision_format import OptionLimits
from max.serve.router._decisions_flow import check_capacity
from max.serve.router._decisions_prompt import QuestionView
from max.serve.router._decisions_scoring import DecisionRequestError
from max.serve.schemas._decisions import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS,
    MAX_SCORE_LEVELS,
    QuestionKind,
)

_SETTINGS = DeciderSettings(
    temperature=1.0,
    temperature_by_type={},
    max_state_tokens=1024,
    isolated_levels=False,
)


def _view(kind: QuestionKind, count: int) -> QuestionView:
    """A question naming ``count`` candidates, with no wording."""
    return QuestionView(
        kind=kind,
        question=None,
        names=[str(index) for index in range(count)],
        details=[None] * count,
    )


def test_chat_limits_name_the_label_alphabets() -> None:
    """The chat wording caps each question at its own label alphabet."""
    assert ChatFormat().limits == OptionLimits(
        choice_options=MAX_CHOICE_OPTIONS,
        score_levels=MAX_SCORE_LEVELS,
        questions=MAX_QUESTIONS,
    )


def test_decider_limits_are_one_letter_per_candidate() -> None:
    """The decider prompt labels ten candidates, choice or score alike."""
    assert MAX_DECIDER_OPTIONS == 10
    assert DeciderFormat(_SETTINGS).limits == OptionLimits(
        choice_options=MAX_DECIDER_OPTIONS,
        score_levels=MAX_DECIDER_OPTIONS,
        questions=MAX_QUESTIONS,
    )


def test_check_capacity_passes_at_the_ceiling() -> None:
    """A question exactly as wide as the format can label is accepted."""
    limits = OptionLimits(
        choice_options=3, score_levels=2, questions=2
    )  # shortest at the ceiling
    check_capacity(
        limits,
        [("a", _view("choice", 3)), ("b", _view("score", 2))],
    )


def test_check_capacity_refuses_a_wider_choice_than_the_alphabet() -> None:
    """One candidate past the letters is refused, and the message names the
    question and the ceiling."""
    limits = OptionLimits(choice_options=3, score_levels=2, questions=None)
    with pytest.raises(
        DecisionRequestError,
        match=r"question 'a'.*at most 3 candidates, this one names 4",
    ):
        check_capacity(limits, [("a", _view("choice", 4))])


def test_check_capacity_refuses_more_questions_than_the_format_answers() -> (
    None
):
    limits = OptionLimits(choice_options=3, score_levels=3, questions=2)
    with pytest.raises(
        DecisionRequestError,
        match=r"at most 2 questions in one request, got 3",
    ):
        check_capacity(
            limits,
            [
                ("a", _view("choice", 1)),
                ("b", _view("choice", 1)),
                ("c", _view("choice", 1)),
            ],
        )


def test_questions_none_places_no_cap_on_the_request() -> None:
    """A format that names no question cap takes any number of them."""
    limits = OptionLimits(choice_options=64, score_levels=64, questions=None)
    check_capacity(limits, [(str(i), _view("choice", 2)) for i in range(200)])
