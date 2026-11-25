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

"""The span table a Locate request states arrives with its context.

``LocateEngine.answer`` names the row's spans on the context the tokenizer
built, and :meth:`Qwen3_5DeciderBatchProcessor.token_positions` cuts the step
into segments from them. The context reaches the model worker as msgpack, and
msgpack writes the fields a dataclass declares: counts set as an extra
attribute stop at the API server, and the worker then counts marker tokens.
"""

from __future__ import annotations

import numpy as np
from max.pipelines.architectures.qwen3_5_decider.batch_processor import (
    attach_segment_bounds,
    segment_bounds,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.modeling.types import (
    RequestID,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)

SPANS = [3, 4, 5]
"""The state block with its separator, then two question chunks."""


def _decode(context: TextContext) -> TextContext:
    """The context as the model worker sees it, past the wire."""
    encode = msgpack_numpy_encoder()
    return msgpack_numpy_decoder(TextContext)(encode(context))


def test_the_stated_spans_cross_to_the_worker() -> None:
    """The worker reads the same counts the request stated."""
    context = TextContext(
        request_id=RequestID("locate-row"),
        max_length=16,
        tokens=TokenBuffer(np.arange(sum(SPANS), dtype=np.int64)),
    )
    attach_segment_bounds(context, SPANS)
    bounds = segment_bounds(_decode(context))
    assert bounds is not None
    assert bounds.spans == tuple(SPANS)
    assert bounds.total == len(context.tokens.active)


def test_a_row_that_states_no_spans_arrives_without_them() -> None:
    """Such a step keeps the batch processor's counting fallback."""
    context = TextContext(
        request_id=RequestID("unstated"),
        max_length=16,
        tokens=TokenBuffer(np.arange(7, dtype=np.int64)),
    )
    assert segment_bounds(_decode(context)) is None
