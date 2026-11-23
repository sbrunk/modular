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

"""``POST /v1/systemone`` over a format that puts two questions on one row.

The route halves are the ones a shared row exercises: ``score_all`` sends one
scoring request per shared prompt, regroups the score rows by marker, and
``usage_totals`` bills the prompt the scorer read once. The fake pipeline records
each call, so the case reads the grouping off the calls and the billing off the
response.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi.testclient import TestClient
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._chat_format import ChatFormat
from max.serve.router._decision_format import OptionLimits
from max.serve.router._decisions_prompt import QuestionScores, QuestionView
from max.serve.router._decisions_scoring import (
    EncodedQuestion,
    EncodedRow,
)
from max.serve.router._systemone_routes import router
from tests.serve.decisions_fakes import (
    MODEL,
    FakePipeline,
    FakeTokenizer,
    make_app,
    prefer_first,
)

_CHOICE = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "charges", "technical": None, "other": None},
}
_SCORE = {
    "type": "score",
    "instructions": "How urgent?",
    "criteria": ["no rush", "soon", "today", "now"],
}


class SharedRowFormat:
    """The chat wording, with every question read at its own marker."""

    def __init__(self) -> None:
        self._chat = ChatFormat()

    @property
    def version(self) -> int:
        """The version of the chat wording this format reuses."""
        return self._chat.version

    @property
    def limits(self) -> OptionLimits:
        """The chat format's option ceilings, unchanged."""
        return self._chat.limits

    async def encode(
        self,
        tokenizer: Any,
        state: Any,
        items: Sequence[tuple[str, QuestionView]],
        chat_template_kwargs: dict[str, Any],
    ) -> list[EncodedQuestion]:
        """Renders each question, then shares one prompt across all of them."""
        rendered = await self._chat.encode(
            tokenizer, state, items, chat_template_kwargs
        )
        shared = rendered[0].rows[0].prompt_ids
        first_marker = len(shared)
        return [
            EncodedQuestion(
                item.question_id,
                item.view,
                [
                    EncodedRow(
                        shared,
                        item.rows[0].label_ids,
                        [first_marker + offset],
                    )
                ],
            )
            for offset, item in enumerate(rendered)
        ]

    def scores(
        self,
        item: EncodedQuestion,
        results: Sequence[LabelScoringOutput],
        temperature: float | None,
    ) -> QuestionScores:
        """Reads one question's option probabilities from its own row."""
        return self._chat.scores(item, results, temperature)


def _post(
    body: dict[str, Any],
    pipeline: FakePipeline,
    **app_kwargs: Any,
) -> Any:
    with TestClient(make_app(pipeline, router, **app_kwargs)) as client:
        return client.post("/v1/systemone", json=body)


def test_two_questions_on_one_row_are_scored_once() -> None:
    """One scoring request, two answers, and the prompt billed once."""
    pipeline = FakePipeline(FakeTokenizer(), prefer_first)
    body = {
        "state": "charged twice",
        "model": MODEL,
        "questions": {"c": _CHOICE, "s": _SCORE},
    }
    response = _post(body, pipeline, decision_format=SharedRowFormat())
    assert response.status_code == 200, response.text
    payload = response.json()
    prompt_length = len(pipeline.prompt_ids[0])
    print(
        f"score calls: {len(pipeline.requests)}; positions: "
        f"{pipeline.positions}; prompt tokens: {prompt_length}; usage: "
        f"{payload['usage']}"
    )
    assert len(pipeline.requests) == 1
    assert pipeline.positions == [[prompt_length, prompt_length + 1]]
    assert payload["usage"]["input_tokens"] == prompt_length
    assert set(payload["answers"]) == {"c", "s"}
    assert payload["answers"]["c"]["choice"] == "billing"
    assert payload["answers"]["s"]["score"] is not None
