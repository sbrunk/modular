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

"""Request and response schemas for ``POST /v1/locate``.

A locate question says what to look for and the answer is where it sits in the
state. Spans are reported by character offset with an exclusive end, so
``state[start:end]`` is the span's text. Offsets count characters, not tokens:
the tokenization that produced the model's rows is the server's own, and a
client holding only the state text can still read a span back.

A locate answer reports no ``probabilities`` and no ``label_mass``. It comes
from the state's hidden rows and the span head's grid over them, where a
``/v1/decisions`` answer comes from next-token scores of candidate labels.
``/v1/systemone`` validates the decisions question union, so locate
questions get their own models.
"""

from __future__ import annotations

from typing import Literal

from max.serve.schemas._decisions import MAX_QUESTIONS
from max.serve.schemas.openai import CompletionUsage
from pydantic import BaseModel, ConfigDict, Field


class LocateQuestion(BaseModel):
    """Wording only: a locate question names no candidates to label."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["locate"] = "locate"
    instructions: str | None = None
    """What to look for. Unset uses the wording the checkpoint ships with."""


class LocateRequest(BaseModel):
    """Body of ``POST /v1/locate``: the state plus questions keyed by id."""

    model_config = ConfigDict(extra="forbid")

    state: str = Field(min_length=1)
    """The text spans are cut from."""
    questions: dict[str, LocateQuestion] = Field(
        min_length=1, max_length=MAX_QUESTIONS
    )
    """Answers come back under the same keys, and the sequence is laid out in
    the request's key order."""


class LocateSpan(BaseModel):
    """A span of the request's state, addressed by character offsets."""

    start: int
    end: int
    """Exclusive, so the span's text is ``state[start:end]``."""
    text: str
    probability: float
    """The span head's probability for this span, after the server's cutoff.

    Named for the quantity: ``DecisionAnswer.score`` is an expected level."""


class LocateAnswer(BaseModel):
    """The spans one question found, in ``(start, end)`` order."""

    spans: list[LocateSpan]
    """Empty when no span cleared the cutoff, which answers "none"."""


class LocateResponse(BaseModel):
    """Response of ``POST /v1/locate``."""

    object: Literal["locate"] = "locate"
    model: str
    cutoff: float
    """Span probabilities below this are dropped before the answer is written."""
    answers: dict[str, LocateAnswer]
    usage: CompletionUsage
    """Nothing is generated, so ``completion_tokens`` is 0."""
