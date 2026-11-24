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

"""Tokenizer and settings fixtures for joint protocol v4 format tests.

The tokenizer is a word-level fake with a controlled vocabulary: the
protocol's marker and separator map to the ids the served architecture
counts, the answer codes map to their configured ids, and every other
piece gets a stable id. The chat template is the shape Qwen3.5's renders
for an open user turn.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from max.pipelines.architectures.qwen3_5_decider.layers.attention import (
    MARKER_TOKEN_ID,
    SEPARATOR_TOKEN_ID,
)
from max.pipelines.modeling.types import (
    TextContentPart,
    TextGenerationRequestMessage,
)
from max.serve.router._decisions_prompt import QuestionView
from max.serve.router._joint_v4_format import (
    JOINT_TEMPLATE,
    MARKER,
    SEPARATOR,
    JointV4Settings,
    parse_joint_v4_settings,
)

CODES = ("c0", "c1", "c2", "c3")
CODE_IDS = (700, 701, 702, 703)

_PIECE = re.compile(r"<\|[a-z_]+\|>|c[0-9]|\S")


def _joint(**overrides: Any) -> dict[str, Any]:
    joint: dict[str, Any] = {
        "version": 4,
        "template": JOINT_TEMPLATE,
        "marker": MARKER,
        "separator": SEPARATOR,
        "max_questions": None,
    }
    joint.update(overrides)
    return joint


def _decision(**overrides: Any) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "codes": list(CODES),
        "token_ids": list(CODE_IDS),
        "temperature": 1.25,
    }
    decision.update(overrides)
    return decision


def _settings(**overrides: Any) -> JointV4Settings:
    return parse_joint_v4_settings(
        _joint(**overrides.pop("joint", {})), _decision(**overrides)
    )


def _choice(*names: str) -> QuestionView:
    names = names or ("billing", "technical", "other")
    return QuestionView(
        kind="choice",
        question=None,
        names=list(names),
        details=[None] * len(names),
    )


def _score(*levels: str) -> QuestionView:
    levels = levels or ("low", "high")
    return QuestionView(
        kind="score",
        question=None,
        names=[str(index) for index in range(len(levels))],
        details=list(levels),
    )


def _yes_no() -> QuestionView:
    return QuestionView(
        kind="yes_no", question=None, names=["yes", "no"], details=[None, None]
    )


class V4Tokenizer:
    """Word-level tokenizer with the joint template's open user turn."""

    eos_token_ids: set[int] = set()

    def __init__(
        self,
        *,
        shift_state: bool = False,
        marker_id: int = MARKER_TOKEN_ID,
        code_ids: dict[str, int] | None = None,
        max_length: int | None = 512,
    ) -> None:
        self._shift_state = shift_state
        if max_length is not None:
            self.max_length = max_length
        self.vocab = {
            MARKER: marker_id,
            SEPARATOR: SEPARATOR_TOKEN_ID,
            **(code_ids or dict(zip(CODES, CODE_IDS))),
        }

    expects_content_wrapping = False

    async def new_context(self, request: Any) -> Any:
        raise NotImplementedError("the fake tokenizer scores without contexts")

    async def decode(self, encoded: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: Any = None,
        **options: Any,
    ) -> str:
        assert options.get("add_generation_prompt") is False
        user = messages[1].content
        assert isinstance(user, list)
        part = user[0]
        if isinstance(part, dict):  # pragma: no cover - defensive
            text = part["text"]
        else:
            assert isinstance(part, TextContentPart)
            text = part.text
        if self._shift_state:
            text = text.replace("\u0000", "\u0000 ")
        return (
            f"<|im_start|>system\n{messages[0].content}<|im_end|>\n"
            f"<|im_start|>user\n{text}"
        )

    def _ids(self, text: str) -> list[int]:
        return [
            self.vocab.get(piece, 1000 + sum(map(ord, piece)))
            for piece in _PIECE.findall(text)
        ]

    async def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        assert add_special_tokens is False
        return np.asarray(self._ids(text), dtype=np.int64)

    async def encode_with_offsets(
        self, text: str, add_special_tokens: bool = False
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """The word-level fake's ids and each piece's character span."""
        assert add_special_tokens is False
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in _PIECE.finditer(text):
            ids.append(
                self.vocab.get(
                    match.group(), 1000 + sum(map(ord, match.group()))
                )
            )
            offsets.append((match.start(), match.end()))
        return ids, offsets
