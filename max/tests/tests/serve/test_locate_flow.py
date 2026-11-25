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

"""The live Locate engine: fake pipeline, real encoder, real span head.

Nothing here touches a GPU. The engine encodes the row with the served
encoder, submits it through a fake pipeline that answers the hidden rows
of the positions it named, runs the real span-head graph on the host over
those rows, and turns the head's grid into spans. The expectations walk
the same path built from the pieces directly, so what is under test is
the wiring between them: which positions are named, how the rows are
split, and what reaches the response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.driver import CPU
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.architectures.qwen3_5_decider.span_head import (
    LOCATE_CONFIG_FILE,
    LOCATE_SUBDIR,
    SpanWeights,
    build_span_head_graph,
    run_span_head,
)
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.request import register_request
from max.serve.router._joint_v4_locate import (
    assemble_spans,
    encode_locate,
)
from max.serve.router._locate_flow import (
    LiveLocateEngine,
    _question_texts,
    locate_router_for_model,
)
from max.serve.router._locate_routes import locate_router
from max.serve.schemas._locate import LocateQuestion, LocateRequest
from safetensors.numpy import save_file
from tests.serve.joint_v4_fakes import V4Tokenizer, _decision, _joint

HIDDEN = 6
WIDTH = 4
PROPOSALS = 6
MAX_SPAN = 4
LOCATE_MARKER_ID = 248077
STATE = "the payments team owns billing-eu"

LOCATE_CONFIG = {
    "format": "pplx-decider-locate-span-v1",
    "head": {
        "max_span": MAX_SPAN,
        "name": "span",
        "proposals": PROPOSALS,
        "width": WIDTH,
    },
    "marker": "<|locate|>",
    "marker_id": LOCATE_MARKER_ID,
    "prompt": "Question:\nLocate verbatim spans:\n{instructions}<|locate|>",
    "prompt_format": "joint-v4-locate",
    "thresholds": {"default": 0.5},
}


def _head_mapping(seed: int = 7) -> dict[str, np.ndarray]:
    """A random head at the saved names, sizes matching the toy settings."""
    rng = np.random.default_rng(seed)
    state = {
        f"{layer}.{part}": rng.standard_normal(
            [WIDTH, HIDDEN] if part == "weight" else [WIDTH]
        ).astype(np.float32)
        for layer in ("ts", "te", "qs", "qe", "qp")
        for part in ("weight", "bias")
    }
    state["thresholds.weight"] = rng.standard_normal([3, HIDDEN]).astype(
        np.float32
    )
    state["thresholds.bias"] = rng.standard_normal([3]).astype(np.float32)
    state["width_bias.weight"] = rng.standard_normal([32, 1]).astype(np.float32)
    state["start_bias"] = np.asarray([0.1], np.float32)
    state["end_bias"] = np.asarray([-0.1], np.float32)
    return state


def _checkpoint(tmp_path: Path, **files: Any) -> Path:
    """Writes a checkpoint the engine accepts, with a tiny head."""
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": HIDDEN}})
    )
    (tmp_path / "joint_config.json").write_text(json.dumps(_joint()))
    (tmp_path / "decision_config.json").write_text(json.dumps(_decision()))
    (tmp_path / LOCATE_SUBDIR).mkdir(exist_ok=True)
    (tmp_path / LOCATE_SUBDIR / LOCATE_CONFIG_FILE).write_text(
        json.dumps(files.get("locate_config", LOCATE_CONFIG))
    )
    if "head" in files:
        save_file(
            files["head"], str(tmp_path / LOCATE_SUBDIR / "head.safetensors")
        )
    return tmp_path


class FakePipeline:
    """A pipeline that answers the named positions with counted rows."""

    model_name = "fake-decider"

    def __init__(self, rows: str = "counted") -> None:
        self.tokenizer = V4Tokenizer()
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    async def score(
        self,
        request: Any,
        label_token_ids: Any,
        label_positions: Any = None,
        label_hidden: bool = False,
        configure_context: Any = None,
    ) -> LabelScoringOutput:
        self.calls.append(
            {
                "prompt": list(request.prompt),
                "label_positions": list(label_positions or []),
                "label_hidden": label_hidden,
            }
        )

        class _Context:
            pass

        if configure_context is not None:
            configure_context(_Context())
        if self.rows == "missing":
            return LabelScoringOutput(
                label_log_probabilities=[],
                prompt_token_count=len(request.prompt),
                label_hidden_rows=None,
            )
        count = len(label_positions or [])
        if self.rows == "short":
            count -= 1
        rows = np.arange(count * HIDDEN, dtype=np.float32).reshape(
            count, HIDDEN
        )
        return LabelScoringOutput(
            label_log_probabilities=[],
            prompt_token_count=len(request.prompt),
            label_hidden_rows=rows,
        )


def _app(engine: Any, pipeline: FakePipeline) -> TestClient:
    app = FastAPI(title="MAX Serve Test")
    register_request(app)
    app.state.pipeline = pipeline
    app.include_router(locate_router(engine))
    return TestClient(app)


def _post(client: TestClient, questions: dict[str, Any]) -> Any:
    return client.post(
        "/v1/locate", json={"state": STATE, "questions": questions}
    )


def test_a_locate_request_answers_with_spans_of_its_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, head=_head_mapping())
    engine = LiveLocateEngine(checkpoint)
    pipeline = FakePipeline()
    with _app(engine, pipeline) as client:
        response = _post(client, {"people": {"type": "locate"}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "locate"
    assert payload["model"] == "fake-decider"
    assert payload["cutoff"] == 0.5
    assert "people" in payload["answers"]
    assert payload["usage"]["completion_tokens"] == 0
    call = pipeline.calls[0]
    assert call["label_hidden"] is True
    assert call["prompt"][-1] == LOCATE_MARKER_ID
    assert len(call["label_positions"]) > 0


def test_the_named_positions_are_the_state_rows_then_the_markers(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, head=_head_mapping())
    engine = LiveLocateEngine(checkpoint)
    pipeline = FakePipeline()
    with _app(engine, pipeline) as client:
        response = _post(
            client,
            {"people": {"type": "locate"}, "errors": {"type": "locate"}},
        )
    assert response.status_code == 200
    call = pipeline.calls[0]
    prompt = call["prompt"]
    named = call["label_positions"]
    marker_positions = [
        index for index, token in enumerate(prompt) if token == LOCATE_MARKER_ID
    ]
    # The last named rows are the two markers, in question order; every
    # earlier named row is a state token, and the state is one run.
    assert named[-2:] == marker_positions
    state_rows = named[:-2]
    assert state_rows == list(
        range(state_rows[0], state_rows[0] + len(state_rows))
    )
    assert marker_positions[0] > state_rows[-1]


def test_the_spans_are_the_heads_over_the_rows_the_engine_asked(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, head=_head_mapping())
    engine = LiveLocateEngine(checkpoint)
    pipeline = FakePipeline()
    with _app(engine, pipeline) as client:
        response = _post(client, {"people": {"type": "locate"}})
    payload = response.json()

    # Recompute the same answer directly: encode, counted rows, head,
    # assembly. Nothing is shared with the engine's own call path except
    # the fixtures.
    tokenizer = V4Tokenizer()
    fmt = engine.decision_format
    separator = asyncio.run(fmt.separator_id(tokenizer))
    row = asyncio.run(
        encode_locate(
            fmt.renderer(tokenizer),
            engine.settings,
            tokenizer,
            STATE,
            ["people"],
            separator,
        )
    )
    rows = np.arange(
        len(row.label_positions()) * HIDDEN, dtype=np.float32
    ).reshape(len(row.label_positions()), HIDDEN)
    session = InferenceSession(devices=[CPU()])
    head = session.load(
        build_span_head_graph(
            hidden_size=HIDDEN,
            width=WIDTH,
            proposals=PROPOSALS,
            max_span=MAX_SPAN,
            device=DeviceRef.CPU(),
        )
    )
    weights = SpanWeights.from_mapping(_head_mapping())
    out = run_span_head(
        head, weights, rows[: row.state_count], rows[row.state_count :], 0.5
    )
    expected = assemble_spans(
        [
            (int(s), int(e), float(p))
            for q, s, e, p in zip(
                out["sel_question"],
                out["sel_start"],
                out["sel_end"],
                out["sel_probability"],
                strict=True,
            )
            if int(q) == 0
        ],
        row.state_char_offsets,
        STATE,
    )
    got = [
        (s["start"], s["end"], s["text"], s["probability"])
        for s in payload["answers"]["people"]["spans"]
    ]
    assert got == [(a, b, t, p) for a, b, t, p in expected]


def test_rows_the_model_cannot_answer_are_501(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, head=_head_mapping())
    engine = LiveLocateEngine(checkpoint)
    for mode in ("missing", "short"):
        pipeline = FakePipeline(rows=mode)
        with _app(engine, pipeline) as client:
            response = _post(client, {"people": {"type": "locate"}})
        assert response.status_code == 501, mode


def test_a_checkpoint_without_a_head_mounts_nothing(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)  # no head.safetensors written
    assert locate_router_for_model(checkpoint) is not None
    # The engine loads; the head tensors are read on first use, and a
    # missing file then raises. A checkpoint without locate settings at
    # all is refused at once:
    (checkpoint / LOCATE_SUBDIR / LOCATE_CONFIG_FILE).unlink()
    assert locate_router_for_model(checkpoint) is None


def test_a_marker_the_graph_does_not_count_mounts_nothing(
    tmp_path: Path,
) -> None:
    bad = dict(LOCATE_CONFIG, marker_id=999)
    checkpoint = _checkpoint(tmp_path, locate_config=bad, head=_head_mapping())
    assert locate_router_for_model(checkpoint) is None


def test_a_decisions_format_that_is_not_joint_v4_mounts_nothing(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, head=_head_mapping())
    (checkpoint / "joint_config.json").unlink()
    assert locate_router_for_model(checkpoint) is None


def test_a_question_naming_no_instructions_speaks_with_its_key() -> None:
    request = LocateRequest(
        state="some state",
        questions={
            "people": LocateQuestion(),
            "errors": LocateQuestion(instructions="  error codes  "),
        },
    )
    names, instructions = _question_texts(request)
    assert names == ["people", "errors"]
    assert instructions == ["people", "  error codes  "]
