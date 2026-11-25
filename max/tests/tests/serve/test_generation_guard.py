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
"""A generative route refuses a checkpoint whose head is a readout.

Sampling readout rows returns row indices read as token ids: HTTP 200 and
punctuation. These tests pin that the guard says so at the route, before the
model runs, and that it stays out of the way of everything else.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from max.serve.router._chat_format import ChatFormat
from max.serve.router._generation_guard import require_generation
from max.serve.router._joint_v4_format import JointV4Format
from tests.serve.joint_v4_fakes import _settings


def _app(decision_format: object | None = None) -> FastAPI:
    """An app with one generative route, guarded like the real ones.

    Args:
        decision_format: What the server detected in the checkpoint, or
            ``None`` to leave the state unset (a non-text task).
    """
    app = FastAPI(title="generation guard test")
    app.state.generated = []

    @app.post("/v1/chat/completions")
    async def generate(request: Request) -> dict[str, str]:
        require_generation(request, "/v1/chat/completions")
        app.state.generated.append("sampled")
        return {"content": "sure"}

    if decision_format is not None:
        app.state.decision_format = decision_format
    return app


def test_a_server_that_declared_no_format_generates() -> None:
    """A pipeline built outside the text-generation startup has no format, so
    the guard has nothing to say."""
    app = _app()
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 200
    assert app.state.generated == ["sampled"]


def test_an_ordinary_chat_model_generates() -> None:
    """The chat format means a checkpoint with no decision protocol: its head
    is the vocabulary the route samples."""
    app = _app(ChatFormat())
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 200
    assert app.state.generated == ["sampled"]


def test_the_readout_checkpoint_is_refused_before_it_runs() -> None:
    """Protocol v4 gets 400 naming the routes that answer, and the sampler is
    never reached."""
    app = _app(JointV4Format(_settings()))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 400
    assert app.state.generated == []
    detail = response.json()["detail"]
    assert "4 answer rows" in detail
    assert "/v1/decisions" in detail
    assert "/v1/locate" in detail
