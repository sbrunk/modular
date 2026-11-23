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

"""A route module an outside package contributes through the settings.

``router`` is the module attribute ``resolve_extra_routers`` reads, and it is
the ``/v1/locate`` router of ``max.serve.router._locate_routes`` over
``StubLocateEngine``. The stub cuts one span out of the request's state, so a
test that needs the mounted route needs no model, no tokenizer and no worker.

A ``state`` beginning ``RAISE`` reaches ``LocateRowsUnavailable`` and answers
HTTP 501: the engine's rows do not cover that request.
"""

from __future__ import annotations

from max.serve.router._decisions_flow import LocateRowsUnavailable
from max.serve.router._locate_routes import locate_router
from max.serve.schemas._locate import (
    LocateAnswer,
    LocateRequest,
    LocateResponse,
    LocateSpan,
)
from max.serve.schemas.openai import CompletionUsage

MODEL = "stub-locate"
"""The model name this engine reports; no checkpoint stands behind it."""


class StubLocateEngine:
    """Answers every question with the first 12 characters of its state."""

    def answer(self, locate_request: LocateRequest) -> LocateResponse:
        if locate_request.state.startswith("RAISE"):
            raise LocateRowsUnavailable(
                "the rows this server holds cover 512 state tokens, this "
                "request encodes to more"
            )
        text = locate_request.state[:12]
        return LocateResponse(
            model=MODEL,
            cutoff=0.5,
            answers={
                question_id: LocateAnswer(
                    spans=[
                        LocateSpan(
                            start=0, end=len(text), text=text, probability=0.83
                        )
                    ]
                )
                for question_id in locate_request.questions
            },
            usage=CompletionUsage(
                completion_tokens=0, prompt_tokens=31, total_tokens=31
            ),
        )


router = locate_router(StubLocateEngine())
