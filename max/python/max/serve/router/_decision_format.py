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

"""Prompt formats for decisions: joint v4, decider or chat.

A format decides how a question becomes scoring rows
(:meth:`DecisionFormat.encode`) and how row scores become option
probabilities (:meth:`DecisionFormat.scores`). The server picks one at
startup from the checkpoint's own files: a model that ships
``joint_config.json`` declaring joint protocol v4 gets the joint readout
format, a model that ships ``decider_config.json`` is a decider, and any
other is served with its chat template.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from max.pipelines.modeling.types import LabelScoringTokenizer
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decisions_prompt import QuestionScores, QuestionView
from max.serve.router._decisions_scoring import EncodedQuestion
from max.serve.schemas._decisions import DecisionText


@dataclass(frozen=True)
class OptionLimits:
    """How many candidates one question may name on a given format.

    The schemas admit a request before the server knows which prompt format
    will answer it, so these are the numbers the request flow compares a
    question against. A format names its own ceiling in this record, and the
    schema's ``max_length`` only bounds what a body may queue.
    """

    choice_options: int
    """Named options of one ``choice`` question."""
    score_levels: int
    """Ordered levels of one ``score`` question."""
    questions: int | None
    """Questions per request, or ``None`` when the format caps none."""


class DecisionFormat(Protocol):
    """How a server renders decision questions and reads their scores."""

    @property
    def version(self) -> int:
        """Version of the prompt wording, reported to callers."""
        ...

    @property
    def limits(self) -> OptionLimits:
        """How wide a question this format can render an answer for.

        The request schemas admit a question before this format is chosen, so
        the request flow checks every question against these limits and turns
        an over-wide one into HTTP 400 naming the question.
        """
        ...

    async def encode(
        self,
        tokenizer: LabelScoringTokenizer,
        state: DecisionText,
        items: Sequence[tuple[str, QuestionView]],
        chat_template_kwargs: dict[str, Any],
    ) -> list[EncodedQuestion]:
        """Renders every question to scoring rows.

        Raises:
            DecisionRequestError: If a question cannot be rendered.
        """
        ...

    def scores(
        self,
        item: EncodedQuestion,
        results: Sequence[LabelScoringOutput],
        temperature: float | None,
    ) -> QuestionScores:
        """Option probabilities and label mass for one scored question."""
        ...


def detect_decision_format(model_path: str) -> DecisionFormat:
    """Picks the format for a checkpoint.

    Args:
        model_path: A local checkpoint directory or a Hub repo id.

    Returns:
        The joint v4 format if the checkpoint ships ``joint_config.json``
        declaring protocol v4, the decider format if it ships
        ``decider_config.json``, else the chat format.
    """
    # ``_chat_format``, ``_decider_format`` and ``_joint_v4_format`` import
    # OptionLimits from this module, so importing them inside the function
    # keeps the dependency one-directional.
    from max.serve.router._chat_format import ChatFormat
    from max.serve.router._decider_format import (
        DeciderFormat,
        find_decider_settings,
    )
    from max.serve.router._joint_v4_format import (
        JointV4Format,
        find_joint_v4_settings,
    )

    joint = find_joint_v4_settings(model_path)
    if joint is not None:
        return JointV4Format(joint)
    decider = find_decider_settings(model_path)
    if decider is None:
        return ChatFormat()
    return DeciderFormat(decider)
