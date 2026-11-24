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

"""The Locate fold's ops, on plain CPU tensors.

The fold's whole case for serving decisions and Locate from one graph is
that a gated-off step is exact: ``with_delta`` must return its base bit
for bit when the gate is zero, and ``token_gate`` must read the gate off
the ``<|locate|>`` token per row. These cases run the ops themselves, on
the CPU device, with no checkpoint and no compiled decoder graph; the
served comparison against the reference is the GPU counterpart.

The ``_read_adapter`` cases feed renamed tensor names through the config's
detection, which is what decides whether a build folds at all.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from max.dtype import DType
from max.pipelines.architectures.qwen3_5_decider.layers.adapter import (
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    ADAPTER_SCALE,
    declare_lora,
    install,
    marker_rows,
    token_gate,
    with_delta,
)
from max.pipelines.architectures.qwen3_5_decider.model_config import (
    LOCATE_MARKER_ID,
    _read_adapter,
)
from max.tensor import Tensor

MARKER = LOCATE_MARKER_ID
HIDDEN = 8
RANK = 64  # the only rank install accepts; ADAPTER_SCALE is fixed against it


def _t(values: np.ndarray[Any, Any]) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(values))


class _ShapeOnly:
    """Stands in for ``WeightData``: ``_read_adapter`` only reads ``shape``."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _state(**tensors: tuple[int, ...]) -> dict[str, Any]:
    return {name: _ShapeOnly(shape) for name, shape in tensors.items()}


def test_a_gated_off_delta_is_the_base_bit_for_bit() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal([5, 16], dtype=np.float32)
    base = x @ rng.standard_normal([24, 16], dtype=np.float32).T
    pair_a = rng.standard_normal([RANK, 16], dtype=np.float32)
    pair_b = rng.standard_normal([24, RANK], dtype=np.float32)

    gated_off = with_delta(
        _t(base),
        _t(x),
        (_t(pair_a), _t(pair_b)),
        _t(np.zeros([5], dtype=np.float32)),
    )
    np.testing.assert_array_equal(np.from_dlpack(gated_off), base)


def test_a_gated_on_delta_is_the_lora_formula() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal([5, 16], dtype=np.float32)
    base = rng.standard_normal([5, 24], dtype=np.float32)
    pair_a = rng.standard_normal([RANK, 16], dtype=np.float32)
    pair_b = rng.standard_normal([24, RANK], dtype=np.float32)
    gate = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)

    got = with_delta(_t(base), _t(x), (_t(pair_a), _t(pair_b)), _t(gate))
    shrunk = (x @ pair_a.T) @ pair_b.T
    want = base + (shrunk * ADAPTER_SCALE * gate[:, None]).astype(np.float32)
    assert ADAPTER_SCALE == ADAPTER_ALPHA / ADAPTER_RANK
    np.testing.assert_allclose(np.from_dlpack(got), want, rtol=1e-4, atol=1e-5)


def test_the_gate_turns_on_whole_rows_not_tokens() -> None:
    # Row 0: a decisions row, no marker. Row 1: a Locate row, marker inside.
    # Row 2: an empty row between boundaries. The gate names rows, so every
    # token of row 1 reads 1.0 and every token of row 0 reads 0.0.
    tokens = np.array([7, 8, 9, MARKER, 5, 6], dtype=np.int32)
    offsets = np.array([0, 3, 6, 6], dtype=np.int32)
    gate = token_gate(_t(tokens), _t(offsets), MARKER)
    np.testing.assert_array_equal(
        np.from_dlpack(gate),
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_a_marker_row_gated_off_keeps_its_base_embedding() -> None:
    embeddings = np.arange(3 * HIDDEN, dtype=np.float32).reshape([3, HIDDEN])
    locate = np.full([HIDDEN], -2.0, dtype=np.float32)
    tokens = np.array([MARKER, 4, MARKER], dtype=np.int32)
    gated_off = marker_rows(
        _t(tokens),
        _t(embeddings),
        _t(locate),
        MARKER,
        _t(np.zeros([3], dtype=np.float32)),
    )
    np.testing.assert_array_equal(np.from_dlpack(gated_off), embeddings)
    gated_on = marker_rows(
        _t(tokens),
        _t(embeddings),
        _t(locate),
        MARKER,
        _t(np.ones([3], dtype=np.float32)),
    )
    want = embeddings.copy()
    want[[0, 2]] = locate
    np.testing.assert_array_equal(np.from_dlpack(gated_on), want)


def test_declare_lora_sizes_the_pair_from_the_base_weight() -> None:
    class _Holder:
        weight = Tensor.zeros([24, 16], dtype=DType.float32)
        lora_a: Tensor
        lora_b: Tensor

    holder = _Holder()
    declare_lora(holder, rank=RANK)
    assert tuple(holder.lora_a.shape) == (RANK, 16)
    assert tuple(holder.lora_b.shape) == (24, RANK)
    assert holder.lora_a.dtype is DType.float32

    class _Vector:
        weight = Tensor.zeros([16], dtype=DType.float32)

    with pytest.raises(ValueError, match="no .* weight"):
        declare_lora(_Vector(), rank=RANK)


def test_install_refuses_a_rank_it_cannot_scale() -> None:
    with pytest.raises(ValueError, match="ADAPTER_ALPHA"):
        install(cast(Any, object()), rank=32, marker_id=MARKER)


def test_detection_reads_rank_and_absence_from_tensor_names() -> None:
    assert _read_adapter(
        _state(**{"layers.0.mlp.up_proj.weight": (16, 8)})
    ) == (
        0,
        MARKER,
    )
    good = _state(
        **{
            "embed_tokens.locate": (HIDDEN,),
            "layers.0.mlp.up_proj.lora_a": (RANK, 8),
            "layers.0.mlp.up_proj.lora_b": (16, RANK),
        }
    )
    assert _read_adapter(good) == (RANK, MARKER)

    two_ranks = _state(
        **{
            "embed_tokens.locate": (HIDDEN,),
            "layers.0.mlp.up_proj.lora_a": (RANK, 8),
            "layers.0.mlp.up_proj.lora_b": (16, RANK),
            "layers.1.mlp.up_proj.lora_a": (32, 8),
            "layers.1.mlp.up_proj.lora_b": (16, 32),
        }
    )
    with pytest.raises(ValueError, match="declares 2 ranks"):
        _read_adapter(two_ranks)

    unpaired = _state(
        **{
            "embed_tokens.locate": (HIDDEN,),
            "layers.0.mlp.up_proj.lora_a": (RANK, 8),
        }
    )
    with pytest.raises(ValueError, match="not both"):
        _read_adapter(unpaired)

    no_marker_row = _state(
        **{
            "layers.0.mlp.up_proj.lora_a": (RANK, 8),
            "layers.0.mlp.up_proj.lora_b": (16, RANK),
        }
    )
    with pytest.raises(ValueError, match="embed_tokens.locate"):
        _read_adapter(no_marker_row)
