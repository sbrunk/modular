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

r"""The Locate span head: a second readout over the ``label_hidden`` rows.

The checkpoint's ``locate-span/head.safetensors`` holds fifteen float32
tensors, about 6.6 MB: five ``[width, hidden]`` projections with biases
(``ts``/``te`` over the state rows, ``qs``/``qe``/``qp`` over the marker
rows), a 32-row ``width_bias`` embedding, a ``[3, hidden]`` ``thresholds``
projection whose column 2 is the learned per-question cutoff, and two
scalar biases. This module lowers that head into one MAX ``Graph`` so the
answer to a Locate question comes from the same pass that answered the
rest of the request: the decoder returns the named rows in the
``label_hidden`` channel, and the head consumes them. No second backbone.

The head is two-stage over each question. Stage one scores every state
token as a span start and, separately, as a span end:

```
start = qs @ ts.T / sqrt(width) + start_bias     # [1, n]
end   = qe @ te.T / sqrt(width) + end_bias       # [1, n]
```

and keeps the ``proposals`` best of each. Stage two re-scores each kept
(start, end) pair jointly:

```
pair  = (ts[si] * tanh(qp)) @ te[ei].T / sqrt(width)
score = pair + start_value + end_value + width_bias[floor(log2 width)]
```

with impossible pairs (``end < start``) and pairs wider than ``max_span``
forced to ``-inf``, and the probability is
``sigmoid(score - thresholds[i, 2])``. The selection is every pair whose
probability reaches the serving threshold. Inclusive token spans out;
the caller maps them to text.

Two numbers are baked in at graph-build time because ``top_k`` needs a
compile-time ``k``: ``proposals`` and, through it, the pad width in
:meth:`_padded_top_k`. The threshold stays an input.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef, Graph, TensorType, ops

from .model_config import LOCATE_MARKER_ID

LOCATE_SUBDIR = "locate-span"
"""The directory beside the shards that holds the Locate tensors."""

LOCATE_CONFIG_FILE = "locate_config.json"
"""The head's settings, written by the release that made the tensors."""

LOCATE_FORMAT = "pplx-decider-locate-span-v1"
"""The ``format`` value a compatible ``locate_config.json`` declares."""

HEAD_TENSOR_NAMES: tuple[str, ...] = (
    *(
        f"{layer}.{part}"
        for layer in ("ts", "te", "qs", "qe", "qp", "thresholds")
        for part in ("weight", "bias")
    ),
    "width_bias.weight",
    "start_bias",
    "end_bias",
)
"""The fifteen tensors of ``head.safetensors``, by saved name."""


@dataclass(frozen=True)
class LocateSettings:
    """What ``locate-span/locate_config.json`` declares."""

    width: int
    proposals: int
    max_span: int
    threshold: float
    marker: str
    marker_id: int
    prompt: str
    prompt_format: str

    @property
    def scale(self) -> float:
        """``sqrt(width)``, the head's logit divisor."""
        return math.sqrt(self.width)


def find_locate_settings(model_path: str | Path) -> LocateSettings | None:
    """Returns the checkpoint's Locate settings, or ``None`` without them.

    A checkpoint without ``locate-span/`` cannot answer Locate questions;
    the architecture serves decisions only and says so in its build log.

    Raises:
        ValueError: If a ``locate_config.json`` exists but names a different
            ``format``, which means its tensors may not match this module's.
    """
    config = Path(model_path) / LOCATE_SUBDIR / LOCATE_CONFIG_FILE
    if not config.is_file():
        return None
    raw = json.loads(config.read_text())
    if raw.get("format") != LOCATE_FORMAT:
        raise ValueError(
            f"{config} declares format {raw.get('format')!r}; this module"
            f" reads {LOCATE_FORMAT!r} tensors and thresholds"
        )
    head = raw["head"]
    marker_id = int(raw["marker_id"])
    if marker_id != LOCATE_MARKER_ID:
        raise ValueError(
            f"{config} names marker_id {marker_id}, and this build counts"
            f" marker {LOCATE_MARKER_ID}: the graph gates the adapter on"
            " that constant and counts it as a chunk end, so a checkpoint"
            " trained on another marker cannot be served without changing"
            " the constant"
        )
    return LocateSettings(
        width=int(head["width"]),
        proposals=int(head["proposals"]),
        max_span=int(head["max_span"]),
        threshold=float(raw["thresholds"]["default"]),
        marker=str(raw["marker"]),
        marker_id=marker_id,
        prompt=str(raw["prompt"]),
        prompt_format=str(raw["prompt_format"]),
    )


@dataclass(frozen=True)
class SpanWeights:
    """The head's tensors as float32 numpy arrays, at the saved names."""

    ts_weight: np.ndarray
    ts_bias: np.ndarray
    te_weight: np.ndarray
    te_bias: np.ndarray
    qs_weight: np.ndarray
    qs_bias: np.ndarray
    qe_weight: np.ndarray
    qe_bias: np.ndarray
    qp_weight: np.ndarray
    qp_bias: np.ndarray
    thresholds_weight: np.ndarray
    thresholds_bias: np.ndarray
    width_bias: np.ndarray
    """``[32]``; the saved tensor is ``[32, 1]`` and the head squeezes it."""

    start_bias: float
    end_bias: float

    @property
    def width(self) -> int:
        return int(self.ts_weight.shape[0])

    @property
    def scale(self) -> float:
        return math.sqrt(self.width)

    @classmethod
    def from_mapping(cls, state: Mapping[str, Any]) -> SpanWeights:
        """Reads the head tensors out of a ``head.safetensors`` mapping.

        Raises:
            ValueError: If any of :data:`HEAD_TENSOR_NAMES` is missing, which
                means the release's head differs from this module's.
        """
        missing = [name for name in HEAD_TENSOR_NAMES if name not in state]
        if missing:
            raise ValueError(f"head state is missing {sorted(missing)}")
        return cls(
            **{
                f"{name}_weight": np.asarray(state[f"{name}.weight"])
                for name in ("ts", "te", "qs", "qe", "qp", "thresholds")
            },
            **{
                f"{name}_bias": np.asarray(state[f"{name}.bias"])
                for name in ("ts", "te", "qs", "qe", "qp", "thresholds")
            },
            width_bias=np.asarray(state["width_bias.weight"]).reshape(-1),
            start_bias=float(np.asarray(state["start_bias"]).reshape(())),
            end_bias=float(np.asarray(state["end_bias"]).reshape(())),
        )

    def feed(self) -> list[np.ndarray]:
        """The fifteen tensors in the graph's input order."""
        return [
            self.ts_weight,
            self.ts_bias,
            self.te_weight,
            self.te_bias,
            self.qs_weight,
            self.qs_bias,
            self.qe_weight,
            self.qe_bias,
            self.qp_weight,
            self.qp_bias,
            self.thresholds_weight,
            self.thresholds_bias,
            self.width_bias,
            np.asarray([self.start_bias], np.float32),
            np.asarray([self.end_bias], np.float32),
        ]


def _constant(value: float, dtype: DType, device: DeviceRef) -> Any:
    """A rank-0 constant, from the Python-literal path of ``ops.constant``: a
    0-d numpy array lands as ``[1]``, and ``pad`` demands a ranked scalar fill.
    """
    return ops.constant(value, dtype, device)


def _bucket_ladder(widths: Any, device: DeviceRef, buckets: int = 32) -> Any:
    """``floor(log2(widths))`` for integers, as a count of comparisons.

    ``floor(log2 w)`` is the number of powers ``2**1 .. 2**31`` that are at
    most ``w``, so summing 31 integer comparisons reproduces it exactly for
    every ``1 <= w <= 2**31`` and returns 31 for any wider width, matching
    the reference's ``clip``. No transcendental is involved, so a float
    ``log2`` that misrounds near a power of two cannot move a bucket. MAX
    exposes no ``log2``.
    """
    one = _constant(1, DType.int64, device)
    clamped = ops.max(widths, one)
    bucket = ops.cast(
        ops.greater_equal(clamped, _constant(2, DType.int64, device)),
        DType.int64,
    )
    for power in range(2, buckets):
        bucket = ops.add(
            bucket,
            ops.cast(
                ops.greater_equal(
                    clamped, _constant(1 << power, DType.int64, device)
                ),
                DType.int64,
            ),
        )
    return bucket


def _padded_top_k(
    logits: Any, pad_rows: Any, proposals: int, device: DeviceRef
) -> tuple[Any, Any, Any]:
    """Top-``proposals`` of ``logits``, padded so ``k`` is a compile-time constant.

    The head takes ``k = min(proposals, n_rows)``, which a static graph
    cannot express. ``logits`` gains ``proposals`` ``-inf`` columns and
    ``pad_rows`` its ``proposals`` zero partner rows: ``-inf`` sorts below
    every finite logit, so the first ``min(proposals, n_rows)`` slots are
    the values the head picks and the rest are ``-inf`` with a partner row
    of zeros. ``is_inf`` flags a pad slot later, so no value for ``n`` is
    ever needed.

    Returns ``(values, indices, partner_rows)``.
    """
    # The fill of a constant pad is a host operand: ``mo.pad.constant``
    # constrains operand 2 to ``cpu:0``, so this constant is built on the
    # CPU even though the tensor it pads sits on the graph's device.
    neg_inf = _constant(float("-inf"), DType.float32, DeviceRef.CPU())
    padded = ops.pad(
        logits, [0, 0, 0, proposals], mode="constant", value=neg_inf
    )
    partners = ops.pad(pad_rows, [0, proposals, 0, 0])
    values, indices = ops.top_k(padded, k=proposals, axis=1)
    return values, indices, ops.gather(partners, indices, axis=0)


GRAPH_INPUT_NAMES: tuple[str, ...] = (
    "document",
    "labels",
    "threshold",
    "ts_weight",
    "ts_bias",
    "te_weight",
    "te_bias",
    "qs_weight",
    "qs_bias",
    "qe_weight",
    "qe_bias",
    "qp_weight",
    "qp_bias",
    "thresholds_weight",
    "thresholds_bias",
    "width_bias",
    "start_bias",
    "end_bias",
)
"""Feed order: the two row operands, the serving cutoff, then the fifteen tensors."""

GRAPH_OUTPUT_NAMES: tuple[str, ...] = (
    "start_index",
    "start_value",
    "end_index",
    "end_value",
    "probabilities",
    "cutoff",
    "sel_question",
    "sel_start",
    "sel_end",
    "sel_probability",
)
"""Dense intermediates first, then the ``[matches]`` selection vectors."""


def build_span_head_graph(
    *,
    hidden_size: int,
    width: int,
    proposals: int,
    max_span: int,
    select: bool = True,
    device: DeviceRef | None = None,
    width_buckets: int = 32,
) -> Graph:
    """Builds the span-head graph.

    Args:
        hidden_size: Row width of ``document`` and ``labels``; the model's
            hidden size, which the head's projections map to ``width``.
        width: Head width (``head.width``), and the divisor ``sqrt(width)``.
        proposals: Start rows and end rows kept per question, baked in as
            the ``top_k`` ``k`` and as the pad width.
        max_span: Pairs wider than this get a ``-inf`` score.
        select: Emit the ``[matches]`` candidate vectors from ``nonzero``.
            With ``False`` the graph stops at the dense
            ``[q, proposals, proposals]`` probabilities.
        device: Where the graph runs; ``DeviceRef.CPU()`` by default.
        width_buckets: Rows of ``width_bias``; the bucket ladder indexes it,
            so a width at or beyond ``2**width_buckets`` lands on the last row.

    Returns:
        A ``Graph`` whose inputs and outputs are named by
        :data:`GRAPH_INPUT_NAMES` and :data:`GRAPH_OUTPUT_NAMES`.
    """
    device = device or DeviceRef.CPU()
    f32 = DType.float32
    weight_shapes = [
        ([width, hidden_size], f32),
        ([width], f32),
        ([width, hidden_size], f32),
        ([width], f32),
        ([width, hidden_size], f32),
        ([width], f32),
        ([width, hidden_size], f32),
        ([width], f32),
        ([width, hidden_size], f32),
        ([width], f32),
        ([3, hidden_size], f32),
        ([3], f32),
        ([width_buckets], f32),
        ([1], f32),
        ([1], f32),
    ]
    input_types = [
        TensorType(f32, ["n", hidden_size], device=device),
        TensorType(f32, ["q", hidden_size], device=device),
        TensorType(f32, [1], device=device),
        *[
            TensorType(dtype, shape, device=device)
            for shape, dtype in weight_shapes
        ],
    ]

    with Graph("locate_span_head", input_types=input_types) as graph:
        (
            document,
            labels,
            threshold,
            ts_weight,
            ts_bias,
            te_weight,
            te_bias,
            qs_weight,
            qs_bias,
            qe_weight,
            qe_bias,
            qp_weight,
            qp_bias,
            thresholds_weight,
            thresholds_bias,
            width_bias,
            start_bias,
            end_bias,
        ) = (value.tensor for value in graph.inputs)

        scale = _constant(math.sqrt(width), f32, device)
        neg_inf = _constant(float("-inf"), f32, device)
        one_i = _constant(1, DType.int64, device)

        def project(rows: Any, weight: Any, bias: Any) -> Any:
            return ops.add(ops.matmul(rows, ops.transpose(weight, 0, 1)), bias)

        ts = project(document, ts_weight, ts_bias)
        te = project(document, te_weight, te_bias)
        qs = project(labels, qs_weight, qs_bias)
        qe = project(labels, qe_weight, qe_bias)
        qp = project(labels, qp_weight, qp_bias)
        thresholds = project(labels, thresholds_weight, thresholds_bias)

        start = ops.add(
            ops.div(ops.matmul(qs, ops.transpose(ts, 0, 1)), scale), start_bias
        )
        end = ops.add(
            ops.div(ops.matmul(qe, ops.transpose(te, 0, 1)), scale), end_bias
        )

        start_value, start_index, ts_rows = _padded_top_k(
            start, ts, proposals, device
        )
        end_value, end_index, te_rows = _padded_top_k(
            end, te, proposals, device
        )

        pair = ops.matmul(
            ops.mul(ts_rows, ops.tanh(qp)[:, None, :]),
            ops.transpose(te_rows, 1, 2),
        )
        widths = ops.add(
            ops.sub(end_index[:, None, :], start_index[:, :, None]), one_i
        )
        bias_rows = ops.gather(
            width_bias, _bucket_ladder(widths, device, width_buckets), axis=0
        )
        offsets = ops.add(
            start_value[:, :, None],
            ops.add(end_value[:, None, :], bias_rows),
        )
        scores = ops.add(ops.div(pair, scale), offsets)

        too_wide = ops.greater(widths, _constant(max_span, DType.int64, device))
        too_narrow = ops.logical_not(ops.greater_equal(widths, one_i))
        scores = ops.where(
            ops.logical_or(too_narrow, too_wide), neg_inf, scores
        )

        cutoff = ops.reshape(ops.slice_tensor(thresholds, [..., 2]), [-1, 1, 1])
        probabilities = ops.sigmoid(ops.sub(scores, cutoff))

        outputs = [
            start_index,
            start_value,
            end_index,
            end_value,
            probabilities,
            ops.reshape(cutoff, [-1]),
        ]
        if select:
            # A pad slot is a slot whose logit came out of the -inf columns.
            # No finite hidden row produces one, so `is_inf` flags it and the
            # graph needs no value for `n`.
            real = ops.logical_and(
                ops.logical_not(ops.is_inf(start_value))[:, None, :],
                ops.logical_not(ops.is_inf(end_value))[:, None, :],
            )
            chosen = ops.logical_and(
                real,
                ops.greater_equal(
                    probabilities, ops.reshape(threshold, [1, 1, 1])
                ),
            )
            multi = ops.nonzero(chosen, out_dim="matches")
            question = ops.slice_tensor(multi, [..., 0])
            row_slot = ops.slice_tensor(multi, [..., 1])
            col_slot = ops.slice_tensor(multi, [..., 2])
            outputs += [
                question,
                ops.gather_nd(
                    start_index, ops.stack([question, row_slot], axis=1)
                ),
                ops.gather_nd(
                    end_index, ops.stack([question, col_slot], axis=1)
                ),
                ops.gather_nd(probabilities, multi),
            ]
        # A graph takes exactly one `graph.output` call: a second call puts an
        # `mo.output` op in the middle of the block and verification fails.
        graph.output(*outputs)
    return graph


def run_span_head(
    model: Model,
    weights: SpanWeights,
    document: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Executes the graph on host arrays and returns numpy outputs by name.

    ``model`` must have loaded a graph from :func:`build_span_head_graph`
    with ``select=True``. The head tensors are re-fed on each call; they are
    6.6 MB, and this stays a per-request cost by design while one process
    owns one head.
    """
    feed = [
        np.ascontiguousarray(document, np.float32),
        np.ascontiguousarray(labels, np.float32),
        np.asarray([threshold], np.float32),
        *(np.ascontiguousarray(a, np.float32) for a in weights.feed()),
    ]
    outputs = model.execute(*[Buffer.from_numpy(a) for a in feed])
    return {
        name: output.to(CPU()).to_numpy()
        for name, output in zip(GRAPH_OUTPUT_NAMES, outputs, strict=True)
    }
