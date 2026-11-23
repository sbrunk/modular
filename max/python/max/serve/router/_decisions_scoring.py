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

"""Shared encode-and-score steps behind ``/v1/decisions`` and ``/v1/systemone``.

A question is rendered into one or more prompts (rows) that each end at an
answer position. One prefill step per row returns the full-vocabulary
log-probability of the row's answer labels (see
:meth:`~max.serve.pipelines.llm.TokenGeneratorPipeline.score`); no token is
generated. The prompt format (:mod:`max.serve.router._decision_format`)
decides how many rows a question takes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from fastapi import Request
from max.pipelines.context import SamplingParams
from max.pipelines.modeling.types import (
    LabelScoringTokenizer,
    PipelineTask,
    RequestID,
    TextGenerationRequest,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.router._decisions_prompt import QuestionView


class DecisionRequestError(ValueError):
    """A request this server or model cannot answer, which is the caller's to fix.

    The routes map it to HTTP 400. Any other ``ValueError`` is a server fault
    and becomes a logged 500.
    """


@dataclass(frozen=True)
class EncodedRow:
    """One prompt to score and the label tokens read at its end."""

    prompt_ids: list[int]
    """The prompt, sent to the scorer as token ids so the labels stay valid
    next-token candidates after exactly this tokenization."""
    label_ids: list[int]
    label_positions: list[int] | None = None
    """Positions inside ``prompt_ids`` to read this row's labels at. Several
    questions can name the same ``prompt_ids`` and each name one position: the
    route scores that prompt once and reads one score row per position. With
    ``None`` the prompt is scored at its last position, alone."""


@dataclass(frozen=True)
class EncodedQuestion:
    """A question rendered to scoring rows."""

    question_id: str
    view: QuestionView
    rows: list[EncodedRow]


def require_scoring(
    request: Request, pipeline: TokenGeneratorPipeline, route: str
) -> LabelScoringTokenizer:
    """Refuses servers and models that cannot score labels.

    Args:
        request: The incoming request, for the app state.
        pipeline: The pipeline serving the request.
        route: The route path, named in the error.

    Returns:
        The pipeline's tokenizer, typed for what scoring needs.

    Raises:
        DecisionRequestError: If the server does not run text generation, uses
            LoRA, variable logits or speculative decoding (whose pipelines
            have no scoring path), or its tokenizer lacks a chat template or a
            context length.
    """
    state = request.app.state
    if state.task != PipelineTask.TEXT_GENERATION:
        raise DecisionRequestError(
            f"{route} requires a text generation model, but this server "
            f"runs {state.task.value}"
        )
    config = state.pipeline_config
    unsupported = [
        feature
        for feature, enabled in (
            ("speculative decoding", config.speculative is not None),
            ("LoRA", config.lora is not None),
            ("variable logits", config.sampling.enable_variable_logits),
        )
        if enabled
    ]
    if unsupported:
        raise DecisionRequestError(
            f"{route} does not support a server using {', '.join(unsupported)}"
        )
    tokenizer = pipeline.tokenizer
    if not isinstance(tokenizer, LabelScoringTokenizer):
        raise DecisionRequestError(
            f"{route} requires a tokenizer with a chat template and a "
            "context length"
        )
    return tokenizer


async def score_all(
    request: Request,
    pipeline: TokenGeneratorPipeline,
    model_name: str,
    encoded: Sequence[EncodedQuestion],
) -> list[list[LabelScoringOutput]]:
    """Scores every row concurrently, cancelling the rest on a failure.

    Rows that share one prompt and name positions are scored by one
    request: the shared row is prefilled once, the readout gathers one logits
    row per named position, and each row gets its position's scores back.

    Returns:
        One list of row results per question, in row order.
    """
    request_id = request.state.request_id
    rows = [row for item in encoded for row in item.rows]
    # A row that names a position belongs to the scoring request of its prompt;
    # a row that names none is scored on its own, keyed by its index.
    groups: dict[tuple[object, object], list[int]] = {}
    for index, row in enumerate(rows):
        key: tuple[object, object] = (
            ("prompt", tuple(row.prompt_ids))
            if row.label_positions is not None
            else ("row", index)
        )
        groups.setdefault(key, []).append(index)

    # Each group is one scoring request: its members share the prompt, and the
    # positions and candidate lists are settled before any task starts so a row
    # that names too many positions is refused without leaving launches behind.
    asked: list[
        tuple[list[int], list[int] | list[list[int]], list[int] | None]
    ] = [
        (
            rows[members[0]].prompt_ids,
            _candidates(rows, members),
            _positions(rows, members),
        )
        for members in groups.values()
    ]
    tasks = [
        asyncio.ensure_future(
            pipeline.score(
                TextGenerationRequest(
                    request_id=RequestID(f"{request_id}_{group}"),
                    model_name=model_name,
                    prompt=prompt,
                    sampling_params=SamplingParams(top_k=1, temperature=0),
                    timestamp_ns=request.state.request_timer.start_ns,
                    request_path=request.url.path,
                ),
                candidates,
                positions,
            )
        )
        for group, (prompt, candidates, positions) in enumerate(asked)
    ]
    try:
        scored = list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    # Hand each row back the score row of the position it named, so a question
    # reads its own answer whatever row it shared.
    flat: list[LabelScoringOutput] = [None] * len(rows)  # type: ignore[list-item]
    for group, members in enumerate(groups.values()):
        result = scored[group]
        for offset, member in enumerate(members):
            rows_out = result.label_log_probability_rows
            flat[member] = LabelScoringOutput(
                label_log_probabilities=(
                    rows_out[offset]
                    if rows_out is not None
                    else result.label_log_probabilities
                ),
                prompt_token_count=result.prompt_token_count,
                cached_token_count=result.cached_token_count,
            )
    results: list[list[LabelScoringOutput]] = []
    start = 0
    for item in encoded:
        results.append(flat[start : start + len(item.rows)])
        start += len(item.rows)
    return results


def _candidates(
    rows: Sequence[EncodedRow], members: Sequence[int]
) -> list[int] | list[list[int]]:
    """What one scoring request is asked to score.

    A request over positions takes one candidate list per position, in row
    order, whatever the count: a group of one member is a positions request
    too, and the scorer wants a list per scored row either way. A request that
    names no position has one row and takes its candidate ids flat.
    """
    if rows[members[0]].label_positions is None:
        return rows[members[0]].label_ids
    return [rows[member].label_ids for member in members]


def _positions(
    rows: Sequence[EncodedRow], members: Sequence[int]
) -> list[int] | None:
    """The positions the members of one scoring request named, in row order.

    ``None`` when none of them named a position, which leaves the scorer on its
    last-prompt-position rule.

    Raises:
        DecisionRequestError: If a row names more than one position. The route
            hands one question the scores of one row, so a row that named two
            would answer it from the first and drop the rest.
    """
    positions: list[int] = []
    for member in members:
        named = rows[member].label_positions
        if named is None:
            return None
        if len(named) != 1:
            raise DecisionRequestError(
                f"row {member} of this request names {len(named)} label "
                f"positions {list(named)}: the decisions route reads one "
                "question at one position, so a row names at most one"
            )
        positions.append(int(named[0]))
    return positions


def usage_totals(
    results: Sequence[Sequence[LabelScoringOutput]],
    encoded: Sequence[EncodedQuestion] | None = None,
) -> tuple[int, int | None]:
    """Total prompt tokens, and prefix-cache hits if the pipeline reports them.

    A prompt shared by several questions is scored once, so its tokens are
    counted once. ``encoded`` names the rows each result belongs to; without
    it every result counts as a row of its own.
    """
    flat = [result for rows in results for result in rows]
    if encoded is None:
        counted = flat
    else:
        rows = [row for item in encoded for row in item.rows]
        counted = []
        shared: set[tuple[int, ...]] = set()
        for row, result in zip(rows, flat, strict=True):
            if row.label_positions is not None:
                key = tuple(row.prompt_ids)
                if key in shared:
                    continue
                shared.add(key)
            counted.append(result)
    prompt_tokens = sum(result.prompt_token_count for result in counted)
    cached = [result.cached_token_count for result in counted]
    cached_tokens = (
        sum(count or 0 for count in cached)
        if any(count is not None for count in cached)
        else None
    )
    return prompt_tokens, cached_tokens
