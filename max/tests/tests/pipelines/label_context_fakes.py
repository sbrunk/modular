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

"""The smallest context a label-scoring rule can be asked of.

``DecoderBatchProcessor.label_positions``, ``BatchLabelScorer.score``,
``BatchLabelScorer.hidden_rows`` and ``greedy_argmax_suffices`` read the label
fields, three token counters and the sampling parameters off a context. The
cases that pin those rules build no tokenizer, token buffer or device.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.pipelines.context import SamplingParams, TextContext


@dataclass
class FakeTokens:
    """The three token counters the row rule reads."""

    active_length: int = 5
    actively_chunked: bool = False
    generated_length: int = 0


def label_context(
    request_id: str,
    label_token_ids: list[int] | None = None,
    label_positions: list[int] | None = None,
    label_token_id_groups: list[list[int]] | None = None,
    label_hidden: bool = False,
    tokens: FakeTokens | None = None,
) -> TextContext:
    """A context naming the label fields, the token counters and greedy
    sampling parameters.

    Built with ``__new__``: a real :class:`TextContext` also wants a tokenizer,
    a token buffer and a grammar state, none of which the label rules read.
    """
    context = TextContext.__new__(TextContext)
    context.request_id = request_id  # type: ignore[assignment]
    context.tokens = tokens or FakeTokens()  # type: ignore[assignment]
    context.sampling_params = SamplingParams(top_k=1, temperature=0)
    context.log_probabilities = 0
    context.log_probabilities_echo = False
    context.label_token_ids = label_token_ids
    context.label_positions = label_positions
    context.label_token_id_groups = label_token_id_groups
    context.label_hidden = label_hidden
    return context
