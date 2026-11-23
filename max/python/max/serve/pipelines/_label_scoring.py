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

"""Output type for scoring candidate labels against one prompt."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelScoringOutput:
    """Result of scoring candidate labels for one prompt."""

    label_log_probabilities: list[float]
    """Full-vocabulary log-probability of each candidate label token, in
    candidate order. It holds the scores of the one row this request scored; a
    request that named several positions scores more than one row, no one of
    which is its own answer, and gets an empty list here and its rows in
    :attr:`label_log_probability_rows`."""
    prompt_token_count: int
    """Number of prompt tokens the model processed."""
    cached_token_count: int | None = None
    """Number of prompt tokens served from the KV prefix cache."""
    label_log_probability_rows: list[list[float]] | None = None
    """Log-probabilities of every row that prompt was scored at, in the order
    the caller named its positions, each with that position's own candidates.
    ``None`` when one row was scored, whose scores are in
    :attr:`label_log_probabilities`."""
