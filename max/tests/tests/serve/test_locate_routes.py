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

"""Route tests for ``POST /v1/locate`` against a fake span engine."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from max.serve.request import register_request
from max.serve.router._locate_routes import locate_router
from max.serve.schemas._locate import (
    LocateAnswer,
    LocateRequest,
    LocateResponse,
    LocateSpan,
)
from max.serve.schemas.openai import CompletionUsage


class FakeEngine:
    """Answers every locate question with the whole state as one span."""

    def __init__(self) -> None:
        self.seen: list[LocateRequest] = []

    async def answer(
        self, locate_request: LocateRequest, request: Request
    ) -> LocateResponse:
        self.seen.append(locate_request)
        state = locate_request.state
        return LocateResponse(
            model="fake-decider",
            cutoff=0.5,
            answers={
                question_id: LocateAnswer(
                    spans=[
                        LocateSpan(
                            start=0,
                            end=len(state),
                            text=state,
                            probability=0.75,
                        )
                    ]
                )
                for question_id in locate_request.questions
            },
            usage=CompletionUsage(
                prompt_tokens=7, completion_tokens=0, total_tokens=7
            ),
        )


def _app(engine: FakeEngine) -> FastAPI:
    app = FastAPI(title="MAX Serve Test")
    register_request(app)
    app.include_router(locate_router(engine))
    return app


def _post(engine: FakeEngine, body: dict[str, Any]) -> Any:
    with TestClient(_app(engine)) as client:
        return client.post("/v1/locate", json=body)


def test_answers_every_question_with_spans_over_the_state() -> None:
    engine = FakeEngine()
    response = _post(
        engine,
        {
            "state": "the queue drained at five",
            "questions": {"when": {"type": "locate"}},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "locate"
    assert payload["model"] == "fake-decider"
    assert payload["cutoff"] == 0.5
    spans = payload["answers"]["when"]["spans"]
    assert spans[0]["text"] == "the queue drained at five"
    assert spans[0]["start"] == 0 and spans[0]["end"] == len(
        "the queue drained at five"
    )
    assert payload["usage"]["completion_tokens"] == 0
    # The engine is handed the parsed request, keyed questions included.
    assert list(engine.seen[0].questions) == ["when"]


def test_an_empty_state_is_refused_before_the_engine_runs() -> None:
    engine = FakeEngine()
    response = _post(
        engine, {"state": "", "questions": {"when": {"type": "locate"}}}
    )
    assert response.status_code == 422
    assert engine.seen == []


def test_a_question_naming_candidates_is_refused() -> None:
    engine = FakeEngine()
    response = _post(
        engine,
        {
            "state": "some state",
            "questions": {"when": {"type": "locate", "options": ["a"]}},
        },
    )
    assert response.status_code == 422
    assert engine.seen == []
