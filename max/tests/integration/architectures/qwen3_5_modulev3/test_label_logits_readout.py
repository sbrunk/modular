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

"""Which rows the Qwen3.5 readout gathers, and who asks for them.

``Qwen3_5TextModel.forward`` returns two readout buffers once a deployment names
positions: ``next_token_logits`` keeps one row per context, the count
``TextGenerationPipeline._sample`` launches the sampler with, and
``label_logits`` returns the rows the host named. ``scored_rows`` decides the
second gather, and an operand-less build gets each sequence's last row, which is
the first gather's rows.

The operand itself is gated: ``label_positions_type`` reads what the decoder
class declares, and the donor ``Qwen3_5Decoder`` declares nothing, so a stock
Qwen3.5 deployment traces the graph it traced before, with one readout and no
``label_positions`` input. ``input_layout`` names the field either way, the same
rule that leaves ``token_positions`` out for a config without M-RoPE.

The readout is staged on the CPU device, and the cases run it over a residual
buffer whose rows are numbered, so an assertion reads row numbers. These
cases load no checkpoint.
"""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import numpy as np
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.qwen3_5_modulev3.qwen3_5 import (
    Qwen3_5Decoder,
    Qwen3_5TextModel,
)
from max.pipelines.lib.module_registration import label_positions_type
from max.pipelines.lib.role_contracts import DecoderInputs
from max.tensor import Tensor

HIDDEN = 4
"""The width of the stand-in residual buffer: column 0 numbers the row."""

ROW_OFFSETS = np.array([0, 6, 10], dtype=np.int64)
"""Two contexts: rows 0..5 and rows 6..9."""

POSITIONS = np.array([2, 4, 9], dtype=np.int64)
"""The scored rows in batch order: the first context named 2 and 4, and the
second named no marker and contributed its last row."""

NEXT_TOKEN_ROWS = [5, 9]
"""One row per context, the shape the sampler launches with."""

DEVICE = DeviceRef.CPU()


class _LabelScoringDecoder(Qwen3_5Decoder):
    """A Qwen3.5 deployment that scores at positions its requests name."""

    label_positions_declared = True


def _numbered_rows(count: int) -> np.ndarray:
    """A residual buffer whose first column is that row's number."""
    return np.arange(count * HIDDEN, dtype=np.float32).reshape(count, HIDDEN)


def _row_numbers(rows: np.ndarray) -> list[int]:
    """The row numbers a gathered block came from."""
    return (rows[:, 0].astype(int) // HIDDEN).tolist()


def _traced_readout(with_operand: bool) -> Graph:
    """Traces the readout: the sampler's rows, and the named rows when asked."""
    input_types = [
        TensorType(DType.float32, ["total_seq_len", HIDDEN], device=DEVICE),
        TensorType(DType.int64, ["batch_plus_1"], device=DEVICE),
    ]
    if with_operand:
        input_types.append(
            TensorType(DType.int64, ["total_label_positions"], device=DEVICE)
        )
    graph = Graph("two_channel_readout", input_types=input_types)
    with graph:
        hidden = graph.inputs[0].tensor
        offsets = graph.inputs[1].tensor
        operand = graph.inputs[2].tensor if with_operand else None
        sampler_rows = F.gather(hidden, offsets[1:] - 1, axis=0)
        sampler_rows.name = "next_token_logits"
        rows = Qwen3_5TextModel.scored_rows(
            cast(Tensor, offsets), cast(Tensor | None, operand)
        )
        gathered = F.gather(hidden, rows, axis=0)
        gathered.name = "label_logits"
        if with_operand:
            graph.output(sampler_rows, gathered)
        else:
            graph.output(sampler_rows)
    return graph


def _gathered_rows(with_operand: bool) -> list[list[int]]:
    """The row numbers of each output leaf of the staged readout."""
    session = InferenceSession(devices=[CPU()])
    model = session.load(_traced_readout(with_operand))
    inputs = [
        Buffer.from_numpy(_numbered_rows(int(ROW_OFFSETS[-1]))).to(CPU()),
        Buffer.from_numpy(ROW_OFFSETS).to(CPU()),
    ]
    if with_operand:
        inputs.append(Buffer.from_numpy(POSITIONS).to(CPU()))
    return [_row_numbers(np.asarray(out.to_numpy())) for out in model(*inputs)]


def test_the_operand_channel_gathers_the_named_rows() -> None:
    """Two contexts, two named markers: three label rows, two sampler rows."""
    assert _gathered_rows(with_operand=True) == [NEXT_TOKEN_ROWS, [2, 4, 9]]


def test_a_build_without_the_operand_gathers_one_row_per_context() -> None:
    """With no operand, ``scored_rows`` returns the next-token rows."""
    assert _gathered_rows(with_operand=False) == [NEXT_TOKEN_ROWS]


def test_the_donor_declares_no_label_operand() -> None:
    """A stock Qwen3.5 deployment gets no ``label_positions`` graph input."""
    assert Qwen3_5Decoder.label_positions_declared is False
    assert "label_positions" in Qwen3_5Decoder.input_layout
    assert label_positions_type(Qwen3_5Decoder, object(), DEVICE) is None


def test_a_label_scoring_deployment_declares_the_operand() -> None:
    """The flag turns the operand into a graph input of the named shape."""
    declared = label_positions_type(_LabelScoringDecoder, object(), DEVICE)
    assert declared is not None
    assert declared.dtype == DType.int64
    assert declared.shape == ["total_label_positions"]
    named = {field.name: field.default for field in fields(DecoderInputs)}
    assert named["label_positions"] is None
