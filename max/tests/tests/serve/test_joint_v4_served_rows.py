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

"""``POST /v1/decisions`` in joint protocol v4: one shared row per request.

The format groups two questions of a request into one prompt that states
the state once, and ``score_all`` scores that prompt once, gathering one
readout row per marker position. This case reads the grouping off the
scoring calls and the per-question answers off the response.
"""

from __future__ import annotations

import math
from typing import Any, cast

from fastapi.testclient import TestClient
from max.pipelines.architectures.qwen3_5_decider.layers.attention import (
    MARKER_TOKEN_ID,
    SEPARATOR_TOKEN_ID,
)
from max.serve.router._decisions_routes import router
from max.serve.router._joint_v4_format import JointV4Format
from tests.serve.decisions_fakes import (
    FakePipeline,
    FakeTokenizer,
    make_app,
)
from tests.serve.joint_v4_fakes import V4Tokenizer, _settings


def _scorer(request: Any, labels: Any) -> list[float]:
    """Answers three-option questions with option 0 and two-option ones with 1."""
    if len(labels) == 3:
        return [math.log(p) for p in (0.7, 0.1, 0.2)]
    return [math.log(p) for p in (0.2, 0.8)]


def _body() -> dict[str, Any]:
    return {
        "input": "a shared state, stated once",
        "questions": [
            {
                "id": "three",
                "type": "choice",
                "question": "Which of three?",
                "options": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
            },
            {
                "id": "two",
                "type": "choice",
                "question": "Which of two?",
                "options": [{"name": "p"}, {"name": "q"}],
            },
        ],
    }


def test_two_questions_score_one_shared_row_read_at_two_markers() -> None:
    # FakePipeline only records the tokenizer; the route hands its own
    # (this one) to the format.
    pipeline = FakePipeline(cast(FakeTokenizer, V4Tokenizer()), _scorer)
    app = make_app(
        pipeline,
        router,
        decision_format=JointV4Format(_settings()),
    )
    with TestClient(app) as client:
        response = client.post("/v1/decisions", json=_body())
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["three"]["choice"] == "x"
    assert body["answers"]["two"]["choice"] == "q"

    # One scoring request carried both questions: the prompt holds the
    # protocol's separator once and the readout marker twice, and the two
    # named positions are exactly those marker rows.
    assert len(pipeline.requests) == 1
    (prompt,) = pipeline.prompt_ids
    assert prompt.count(SEPARATOR_TOKEN_ID) == 1
    markers = [
        index for index, token in enumerate(prompt) if token == MARKER_TOKEN_ID
    ]
    assert len(markers) == 2
    assert pipeline.positions == [sorted(markers)]
    assert pipeline.labels == [[0, 1, 2], [0, 1]]
