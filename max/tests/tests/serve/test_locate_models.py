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

"""Validation tests for the ``POST /v1/locate`` request and response models."""

from __future__ import annotations

import pytest
from max.serve.schemas._decisions import MAX_QUESTIONS
from max.serve.schemas._locate import (
    LocateAnswer,
    LocateQuestion,
    LocateRequest,
    LocateResponse,
    LocateSpan,
)
from max.serve.schemas.openai import CompletionUsage
from pydantic import ValidationError


def _request(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "state": "the incident timeline, in order",
        "questions": {"first": {"type": "locate"}},
    }
    body.update(overrides)
    return body


def test_a_well_formed_request_validates_and_defaults_its_type() -> None:
    request = LocateRequest.model_validate(_request())
    assert request.questions["first"].type == "locate"
    assert request.state


def test_an_empty_state_is_refused() -> None:
    with pytest.raises(ValidationError):
        LocateRequest.model_validate(_request(state=""))


def test_a_question_naming_candidates_is_refused() -> None:
    """Locate wording names no candidates; those belong to ``/v1/decisions``."""
    with pytest.raises(ValidationError):
        LocateRequest.model_validate(
            _request(questions={"first": {"type": "locate", "options": ["a"]}})
        )


def test_an_unknown_question_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        LocateRequest.model_validate(
            _request(questions={"first": {"type": "choice"}})
        )


def test_the_request_makes_at_least_one_question_and_queues_a_bounded_many() -> (
    None
):
    with pytest.raises(ValidationError):
        LocateRequest.model_validate(_request(questions={}))
    too_many = {
        str(index): {"type": "locate"} for index in range(MAX_QUESTIONS + 1)
    }
    with pytest.raises(ValidationError):
        LocateRequest.model_validate(_request(questions=too_many))


def test_extra_keys_are_refused_at_both_levels() -> None:
    for body in (
        _request(temperature=0.0),
        _request(questions={"first": {"type": "locate", "cutoff": 0.5}}),
    ):
        with pytest.raises(ValidationError):
            LocateRequest.model_validate(body)


def test_a_response_reports_spans_and_zero_completion() -> None:
    """The response carries character spans and no generated tokens."""
    response = LocateResponse(
        model="pplx-decider",
        cutoff=0.5,
        answers={
            "first": LocateAnswer(
                spans=[LocateSpan(start=0, end=3, text="the", probability=0.9)]
            ),
            "none": LocateAnswer(spans=[]),
        },
        usage=CompletionUsage(
            prompt_tokens=7, completion_tokens=0, total_tokens=7
        ),
    )
    payload = response.model_dump(exclude_none=True)
    assert payload["object"] == "locate"
    assert payload["answers"]["first"]["spans"][0]["text"] == "the"
    assert payload["answers"]["none"]["spans"] == []
    assert payload["usage"]["completion_tokens"] == 0


def test_a_locate_question_stays_a_wording_only_record() -> None:
    assert LocateQuestion().type == "locate"
    assert LocateQuestion(
        instructions="find the outage window"
    ).instructions == ("find the outage window")
