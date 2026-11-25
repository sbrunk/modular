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

"""Answers ``/v1/locate`` from the pipeline that is already running.

The checkpoint's span head is a second, tiny readout: fifteen float32
tensors, 6.6 MB, scoring the state's hidden rows against each question's
``<|locate|>`` row. This engine answers from the serving pipeline rather
than building a second one: it encodes the row (see
:mod:`max.serve.router._joint_v4_locate`), submits it as a scoring step
whose named positions are the state tokens and the markers, asks the step
for the hidden rows themselves (:attr:`~max.pipelines.context.TextContext
.label_hidden`) instead of candidate scores, and runs the head graph over
those rows on the host. One backbone, one forward pass per request, and
the answer is text the caller can cut out of its own state.

The engine is mounted automatically when the served checkpoint declares
the span head (``locate-span/locate_config.json`` beside the shards and a
joint protocol v4 decisions format); see
:func:`locate_router_for_model`, called from the server's startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from max.driver import CPU
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.architectures.qwen3_5_decider.batch_processor import (
    attach_segment_bounds,
)
from max.pipelines.architectures.qwen3_5_decider.span_head import (
    LOCATE_CONFIG_FILE,
    LOCATE_SUBDIR,
    SpanWeights,
    build_span_head_graph,
    find_locate_settings,
    run_span_head,
)
from max.pipelines.context import SamplingParams
from max.pipelines.modeling.types import (
    LabelScoringTokenizer,
    RequestID,
    TextGenerationRequest,
)
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.router._decision_format import detect_decision_format
from max.serve.router._decisions_flow import LocateRowsUnavailable
from max.serve.router._decisions_scoring import DecisionRequestError
from max.serve.router._joint_v4_format import JointV4Format
from max.serve.router._joint_v4_locate import assemble_spans, encode_locate
from max.serve.router._locate_routes import locate_router
from max.serve.schemas._locate import (
    LocateAnswer,
    LocateRequest,
    LocateResponse,
    LocateSpan,
)
from max.serve.schemas.openai import CompletionUsage
from safetensors.numpy import load_file

logger = logging.getLogger("max.serve")

__all__ = ["LiveLocateEngine", "locate_router_for_model"]


class LiveLocateEngine:
    """Answers Locate requests from the server's own pipeline and tokenizer.

    The span-head session is built once, on the first request: a server
    that never answers Locate never compiles the head. The head runs on
    the host, where its rows arrive from the step; a lock keeps two
    concurrent requests off the one compiled graph, which answers locate
    rows one at a time anyway (the architecture pins a batch of one).
    """

    def __init__(self, model_path: str | Path) -> None:
        """Reads the checkpoint's Locate settings.

        Args:
            model_path: The directory being served, holding ``config.json``
                and ``locate-span/``.

        Raises:
            ValueError: If the directory is not one this engine can answer
                for: no ``locate-span/``, a head this module does not read,
                or a decisions format that is not joint protocol v4.
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ValueError(f"{self.model_path} is not a local checkpoint")
        decision_format = detect_decision_format(str(self.model_path))
        if not isinstance(decision_format, JointV4Format):
            raise ValueError(
                f"{self.model_path} answers decisions with "
                f"{type(decision_format).__name__}; /v1/locate reads the "
                "joint protocol v4 Locate rows and their span head"
            )
        settings = find_locate_settings(self.model_path)
        if settings is None:
            raise ValueError(
                f"{self.model_path / LOCATE_SUBDIR / LOCATE_CONFIG_FILE}"
                " is absent: this checkpoint ships no Locate span head"
            )
        self.decision_format = decision_format
        self.settings = settings
        self._head_lock = threading.Lock()
        self._head_model: Any = None
        self._head_weights: SpanWeights | None = None
        self._separator_id: int | None = None

    async def answer(
        self, locate_request: LocateRequest, request: Request
    ) -> LocateResponse:
        """Answers one Locate request with spans over its state.

        Args:
            locate_request: The validated request body.
            request: The incoming HTTP request, for the app state, the
                request id and the timing base.

        Raises:
            DecisionRequestError: If this server or tokenizer cannot score
                rows, or the row does not fit the context length.
            LocateRowsUnavailable: If the model returned no hidden rows or
                a different count than the row named.
        """
        pipeline: TokenGeneratorPipeline | None = getattr(
            request.app.state, "pipeline", None
        )
        if pipeline is None:
            raise DecisionRequestError(
                "/v1/locate answers from a local pipeline; this host serves "
                "a remote engine"
            )
        tokenizer = pipeline.tokenizer
        if not isinstance(tokenizer, LabelScoringTokenizer) or not hasattr(
            tokenizer, "encode_with_offsets"
        ):
            raise DecisionRequestError(
                "/v1/locate requires a tokenizer with a chat template, a "
                "context length, and offset mapping"
            )
        renderer = self.decision_format.renderer(tokenizer)
        if self._separator_id is None:
            self._separator_id = await self.decision_format.separator_id(
                tokenizer
            )
        state = locate_request.state
        names, instructions = _question_texts(locate_request)
        row = await encode_locate(
            renderer,
            self.settings,
            tokenizer,
            state,
            instructions,
            self._separator_id,
        )
        named = row.label_positions()
        scored = await pipeline.score(
            TextGenerationRequest(
                request_id=RequestID(f"{request.state.request_id}_0"),
                model_name=pipeline.model_name,
                prompt=list(row.prompt_ids),
                sampling_params=SamplingParams(top_k=1, temperature=0),
                timestamp_ns=request.state.request_timer.start_ns,
                request_path="/v1/locate",
            ),
            None,
            named,
            label_hidden=True,
            configure_context=lambda context: attach_segment_bounds(
                context, row.segments
            ),
        )
        hidden = scored.label_hidden_rows
        if hidden is None:
            raise LocateRowsUnavailable(
                "the model returned no label_hidden channel; answering "
                "locate needs an architecture that declares it"
            )
        if hidden.shape[0] != len(named):
            raise LocateRowsUnavailable(
                f"the step returned {hidden.shape[0]} hidden rows for the "
                f"{len(named)} positions {named} the row named"
            )
        head = await asyncio.to_thread(
            self._run_head,
            hidden[: row.state_count],
            hidden[row.state_count :],
        )
        answers: dict[str, LocateAnswer] = {}
        for index, name in enumerate(names):
            candidates = [
                (int(start), int(end), float(probability))
                for question, start, end, probability in zip(
                    head["sel_question"],
                    head["sel_start"],
                    head["sel_end"],
                    head["sel_probability"],
                    strict=True,
                )
                if int(question) == index
            ]
            spans = assemble_spans(candidates, row.state_char_offsets, state)
            answers[name] = LocateAnswer(
                spans=[
                    LocateSpan(
                        start=start, end=end, text=text, probability=score
                    )
                    for start, end, text, score in spans
                ]
            )
        return LocateResponse(
            model=pipeline.model_name,
            cutoff=self.settings.threshold,
            answers=answers,
            usage=CompletionUsage(
                prompt_tokens=scored.prompt_token_count,
                completion_tokens=0,
                total_tokens=scored.prompt_token_count,
            ),
        )

    def _run_head(
        self, document: Any, labels: Any
    ) -> dict[str, Any]:  # numpy outputs by name
        """Runs the span head off the event loop, loading it once."""
        with self._head_lock:
            if self._head_model is None:
                self._load_head()
            assert self._head_weights is not None
            return run_span_head(
                self._head_model,
                self._head_weights,
                document,
                labels,
                self.settings.threshold,
            )

    def _load_head(self) -> None:
        """Compiles the head graph and reads its tensors; once per server."""
        config = json.loads((self.model_path / "config.json").read_text())
        text_config = config.get("text_config", config)
        hidden_size = int(text_config["hidden_size"])
        session = InferenceSession(devices=[CPU()])
        self._head_model = session.load(
            build_span_head_graph(
                hidden_size=hidden_size,
                width=self.settings.width,
                scale=self.settings.scale,
                proposals=self.settings.proposals,
                max_span=self.settings.max_span,
                device=DeviceRef.CPU(),
            )
        )
        weights_file = self.model_path / LOCATE_SUBDIR / "head.safetensors"
        self._head_weights = SpanWeights.from_mapping(load_file(weights_file))
        logger.info(
            "locate span head loaded on the host: width %d, %d proposals, "
            "cutoff %.3g",
            self.settings.width,
            self.settings.proposals,
            self.settings.threshold,
        )


def _question_texts(
    locate_request: LocateRequest,
) -> tuple[list[str], list[str]]:
    """Returns the question keys and their instruction texts, in order.

    A Locate question may name no instructions; the reference then speaks
    for it with its own key, and the rows have to match the reference.
    """
    names = list(locate_request.questions)
    instructions: list[str] = []
    for name in names:
        text = locate_request.questions[name].instructions
        instructions.append(text if text is not None and text.strip() else name)
    return names, instructions


def locate_router_for_model(model_path: str | Path) -> APIRouter | None:
    """Builds the ``/v1/locate`` router for one checkpoint, or ``None``.

    Called at server startup with the checkpoint being served. A checkpoint
    that ships no span head, or answers decisions in another protocol, is
    a server without the route rather than a broken one, so it returns
    ``None`` and says why at debug level.

    Args:
        model_path: The local checkpoint directory.

    Returns:
        A router for ``locate_router``, its engine already bound, or
        ``None`` when this checkpoint cannot answer Locate.
    """
    try:
        engine = LiveLocateEngine(model_path)
    except (ValueError, OSError, KeyError) as e:
        logger.debug(
            "/v1/locate not mounted for %s: %s",
            model_path,
            e,
        )
        return None
    logger.info(
        "/v1/locate mounted for the span head at %s: marker id %d, cutoff %.3g",
        model_path,
        engine.settings.marker_id,
        engine.settings.threshold,
    )
    return locate_router(engine)
