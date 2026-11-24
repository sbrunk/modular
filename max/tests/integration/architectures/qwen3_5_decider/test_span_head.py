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

"""The span head graph against a numpy transcription of the reference head.

``build_span_head_graph`` lowers the released head (five projections, the
width-bucket embedding, the learned cutoff) into one graph. The oracle
here transcribes the same math in numpy at the same float32 precision,
row by row, with a plain ``argsort`` where the graph pads and takes a
constant ``top_k``: the two paths share no code, so agreement is the
point of the case. Everything runs on the CPU device with a toy hidden
size; the served spans against the release oracle are the GPU-side
counterpart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU
from max.engine import InferenceSession
from max.pipelines.architectures.qwen3_5_decider.span_head import (
    GRAPH_INPUT_NAMES,
    GRAPH_OUTPUT_NAMES,
    LocateSettings,
    SpanWeights,
    build_span_head_graph,
    find_locate_settings,
    run_span_head,
)

HIDDEN = 6
WIDTH = 4
QUESTIONS = 2
DOCUMENT_ROWS = 5
PROPOSALS = 6
MAX_SPAN = 4
THRESHOLD = 0.5
"""A case where ``DOCUMENT_ROWS < PROPOSALS``: every slot past the fifth is a
pad slot, and ``MAX_SPAN`` is shorter than the widest pair, so both mask
paths run."""


def _weights(seed: int) -> SpanWeights:
    rng = np.random.default_rng(seed)
    return SpanWeights(
        **{
            f"{layer}_{part}": rng.standard_normal(
                [WIDTH, HIDDEN] if part == "weight" else [WIDTH]
            ).astype(np.float32)
            for layer in ("ts", "te", "qs", "qe", "qp")
            for part in ("weight", "bias")
        },
        thresholds_weight=rng.standard_normal([3, HIDDEN]).astype(np.float32),
        thresholds_bias=rng.standard_normal([3]).astype(np.float32),
        width_bias=rng.standard_normal([32]).astype(np.float32),
        start_bias=float(rng.standard_normal()),
        end_bias=float(rng.standard_normal()),
    )


def _case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    document = rng.standard_normal([DOCUMENT_ROWS, HIDDEN]).astype(np.float32)
    labels = rng.standard_normal([QUESTIONS, HIDDEN]).astype(np.float32)
    return document, labels


def _oracle_candidates(
    weights: SpanWeights,
    document: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, int, float]]:
    """``(question, start_token, end_token, probability)`` of every kept pair.

    The reference math straight: projections, two top-k's over the document
    rows, the joint pair score with the width bucket, the impossible and
    over-wide masks, the learned cutoff, the serving threshold. Slot order
    is descending top-k value, iterated question-major, which is also the
    order the graph's ``nonzero`` reports.
    """
    scale = math.sqrt(WIDTH)

    def project(rows: np.ndarray, name: str) -> np.ndarray:
        return rows @ getattr(weights, f"{name}_weight").T + getattr(
            weights, f"{name}_bias"
        )

    ts = project(document, "ts")
    te = project(document, "te")
    qs = project(labels, "qs")
    qe = project(labels, "qe")
    qp = project(labels, "qp")
    thresholds = project(labels, "thresholds")
    start = qs @ ts.T / scale + weights.start_bias
    end = qe @ te.T / scale + weights.end_bias

    found: list[tuple[int, int, int, float]] = []
    for question in range(QUESTIONS):
        k = min(PROPOSALS, DOCUMENT_ROWS)
        si = np.argsort(-start[question])[:k]
        ei = np.argsort(-end[question])[:k]
        sv, ev = start[question][si], end[question][ei]
        scores = (ts[si] * np.tanh(qp[question])) @ te[ei].T / scale
        widths = ei[None, :] - si[:, None] + 1
        buckets = np.clip(
            np.floor(np.log2(np.clip(widths, 1, None).astype(np.float64))),
            0,
            31,
        ).astype(np.int64)
        scores = (
            scores + sv[:, None] + ev[None, :] + weights.width_bias[buckets]
        )
        scores = np.where((widths <= 0) | (widths > MAX_SPAN), -np.inf, scores)
        probabilities = 1.0 / (
            1.0 + np.exp(-(scores - thresholds[question, 2]))
        )
        rows, cols = np.nonzero(probabilities >= threshold)
        for row, col in zip(rows, cols, strict=True):
            found.append(
                (
                    int(question),
                    int(si[row]),
                    int(ei[col]),
                    float(probabilities[row, col]),
                )
            )
    return found


def _run(seed: int) -> tuple[dict[str, np.ndarray], SpanWeights]:
    weights = _weights(seed)
    document, labels = _case(seed + 1)
    session = InferenceSession(devices=[CPU()])
    graph = build_span_head_graph(
        hidden_size=HIDDEN,
        width=WIDTH,
        proposals=PROPOSALS,
        max_span=MAX_SPAN,
    )
    model = session.load(graph)
    return run_span_head(model, weights, document, labels, THRESHOLD), weights


def test_the_graph_answers_where_the_numpy_head_answers() -> None:
    run, weights = _run(seed=11)
    document, labels = _case(12)
    oracle = _oracle_candidates(weights, document, labels, THRESHOLD)
    assert oracle, "the case must keep at least one pair for the test to mean"
    graph_answer = [
        (int(q), int(s), int(e), float(p))
        for q, s, e, p in zip(
            run["sel_question"],
            run["sel_start"],
            run["sel_end"],
            run["sel_probability"],
            strict=True,
        )
    ]
    assert len(graph_answer) == len(oracle)
    for got, want in zip(graph_answer, oracle, strict=True):
        assert got[:3] == want[:3]
        assert got[3] == pytest.approx(want[3], rel=1e-4, abs=1e-5)


def test_no_selected_pair_exceeds_max_span_or_reaches_a_pad_slot() -> None:
    run, _ = _run(seed=23)
    for question, start, end in zip(
        run["sel_question"], run["sel_start"], run["sel_end"], strict=True
    ):
        assert int(start) < DOCUMENT_ROWS and int(end) < DOCUMENT_ROWS
        assert 0 < int(end) - int(start) + 1 <= MAX_SPAN
    assert np.isneginf(run["start_value"][:, DOCUMENT_ROWS:]).all()


def test_the_learned_cutoff_is_the_third_threshold_column() -> None:
    run, weights = _run(seed=31)
    document, labels = _case(32)
    expected = labels @ weights.thresholds_weight.T + weights.thresholds_bias
    np.testing.assert_allclose(
        run["cutoff"], expected[:, 2], rtol=1e-5, atol=1e-6
    )


def test_feed_and_leaf_names_match_the_built_graph() -> None:
    assert len(GRAPH_INPUT_NAMES) == 18
    selection = GRAPH_OUTPUT_NAMES[6:]
    assert selection == (
        "sel_question",
        "sel_start",
        "sel_end",
        "sel_probability",
    )


def test_settings_come_from_the_release_files(tmp_path: Path) -> None:
    assert find_locate_settings(tmp_path) is None
    config = {
        "format": "pplx-decider-locate-span-v1",
        "head": {"width": WIDTH, "proposals": PROPOSALS, "max_span": MAX_SPAN},
        "thresholds": {"default": 0.5},
        "marker": "<|locate|>",
        "marker_id": 248077,
        "prompt": "Question:\nLocate verbatim spans:\n{instructions}<|locate|>",
        "prompt_format": "joint-v4-locate",
    }
    directory = tmp_path / "locate-span"
    directory.mkdir()
    (directory / "locate_config.json").write_text(json.dumps(config))
    settings = find_locate_settings(tmp_path)
    assert settings is not None
    assert settings.marker_id == 248077
    assert settings.scale == math.sqrt(WIDTH)
    assert isinstance(settings, LocateSettings)

    (directory / "locate_config.json").write_text(
        json.dumps(config | {"format": "pplx-decider-locate-span-v99"})
    )
    with pytest.raises(ValueError, match="pplx-decider-locate-span-v1"):
        find_locate_settings(tmp_path)


def test_weights_read_and_refuse_the_saved_names() -> None:
    weights = _weights(5)
    state = {
        f"{layer}.{part}": getattr(weights, f"{layer}_{part}")
        for layer in ("ts", "te", "qs", "qe", "qp", "thresholds")
        for part in ("weight", "bias")
    }
    with pytest.raises(ValueError, match="missing"):
        SpanWeights.from_mapping(state)
    full = state | {
        "width_bias.weight": np.zeros([32, 1], np.float32),
        "start_bias": np.zeros([1], np.float32),
        "end_bias": np.zeros([1], np.float32),
    }
    loaded = SpanWeights.from_mapping(full)
    assert loaded.width_bias.shape == (32,)
    assert loaded.start_bias == 0.0
    assert loaded.width == WIDTH


def test_the_dense_probabilities_keep_every_proposal_slot() -> None:
    run, _ = _run(seed=41)
    assert run["probabilities"].shape == (QUESTIONS, PROPOSALS, PROPOSALS)
