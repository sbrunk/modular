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

"""Chunk-end counting: the decisions marker and the Locate marker alike.

A step that states no span table reads its segment labels off the row's
tokens, and one step cannot be told which dialect the row speaks: the
decisions renderer closes a chunk with ``<|fim_suffix|>`` and the Locate
renderer with ``<|locate|>``. These cases pin that counting either id
labels the two dialects identically, and that a checkpoint naming another
marker id is refused rather than silently miscounted. No graph and no
checkpoint; the served Locate rows are the GPU-side counterpart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from max.dtype import DType
from max.pipelines.architectures.qwen3_5_decider.batch_processor import (
    segment_labels,
)
from max.pipelines.architectures.qwen3_5_decider.layers.attention import (
    CHUNK_MARKER_IDS,
    MARKER_TOKEN_ID,
    SEPARATOR_TOKEN_ID,
    marker_segments,
)
from max.pipelines.architectures.qwen3_5_decider.model_config import (
    LOCATE_MARKER_ID,
)
from max.pipelines.architectures.qwen3_5_decider.span_head import (
    LOCATE_CONFIG_FILE,
    LOCATE_SUBDIR,
    find_locate_settings,
)
from max.tensor import Tensor

"""One state token, the separator, then two chunks of two tokens each, the
second and fourth of which are markers in either dialect."""
ROW = [
    700,
    SEPARATOR_TOKEN_ID,
    701,
    MARKER_TOKEN_ID,
    702,
    MARKER_TOKEN_ID,
]
LOCATE_ROW = [
    700,
    SEPARATOR_TOKEN_ID,
    701,
    LOCATE_MARKER_ID,
    702,
    LOCATE_MARKER_ID,
]
LABELS = [-1, -1, 0, 0, 1, 1]


def _t(values: list[int]) -> Tensor:
    return Tensor.from_dlpack(
        np.ascontiguousarray(np.asarray(values, dtype=np.int32))
    )


def test_counting_the_pair_labels_both_dialects_alike() -> None:
    offsets = _t([0, len(ROW)])
    decisions = np.from_dlpack(
        marker_segments(_t(ROW), offsets, marker_id=CHUNK_MARKER_IDS).cast(
            DType.int32
        )
    )
    locate = np.from_dlpack(
        marker_segments(
            _t(LOCATE_ROW), offsets, marker_id=CHUNK_MARKER_IDS
        ).cast(DType.int32)
    )
    np.testing.assert_array_equal(decisions, LABELS)
    np.testing.assert_array_equal(locate, LABELS)


def test_counting_one_dialect_blinds_the_other() -> None:
    """Why the pair is the default: one id mislabels the other dialect."""
    offsets = _t([0, len(LOCATE_ROW)])
    blind = np.from_dlpack(
        marker_segments(_t(LOCATE_ROW), offsets).cast(DType.int32)
    )
    # No marker of that dialect in the row: one span covers the questions,
    # which is the mislabeling a Locate row used to hit.
    np.testing.assert_array_equal(blind, [-1, -1, 0, 0, 0, 0])


def test_the_host_twin_agrees_with_the_graph_rule() -> None:
    """``segment_labels`` is the host path of the batch processor fallback."""
    for row in (ROW, LOCATE_ROW):
        host = segment_labels(np.asarray(row, dtype=np.int32), CHUNK_MARKER_IDS)
        np.testing.assert_array_equal(host, LABELS)


def test_a_checkpoint_naming_another_marker_is_refused(
    tmp_path: Path,
) -> None:
    """The count and the gate read a constant, so a foreign id is refused."""
    config = tmp_path / LOCATE_SUBDIR / LOCATE_CONFIG_FILE
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "format": "pplx-decider-locate-span-v1",
                "head": {
                    "width": 4,
                    "proposals": 4,
                    "max_span": 4,
                },
                "marker": "<|other|>",
                "marker_id": 999999,
                "prompt": "{instructions}<|other|>",
                "prompt_format": "joint-v4-locate",
                "thresholds": {"default": 0.5},
            }
        )
    )
    with pytest.raises(ValueError, match="marker_id 999999"):
        find_locate_settings(tmp_path)
