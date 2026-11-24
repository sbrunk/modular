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

r"""Weight rules: the donor's Qwen3.5 mapping plus this checkpoint's two extras.

The checkpoint is the released multimodal Qwen3.5 bundle. Its text tower sits
under ``language_model.``, its vision tower under ``visual.``, its decision
head is the single BF16 tensor ``weight`` in ``readout.safetensors``, and
``locate-span/`` holds an unmerged LoRA adapter and a span head that the
``**/*.safetensors`` weight glob picks up next to the shards.

:func:`max.pipelines.architectures.qwen3_5_modulev3.weight_adapters.qwen3_5_weight_rules`
is reused rather than rewritten; it covers the ``mtp.`` head and the
``model.language_model.`` / ``model.visual.`` nesting of the checkpoints the
donor serves. Two rules are added for the shape this checkpoint actually uses:

- ``language_model.`` is the text tower's only wrapper here (there is no
  ``model.`` above it), and the registered decoder's parameters carry no
  wrapper because the text model is name-transparent.
- ``weight`` becomes the module path of the readout, so the head is a module
  the graph declares instead of a tensor the graph reaches for.

No ``Ignore`` rule is added for the vision tower or the Locate tensors: they
load, match no parameter, and the coverage audit in :mod:`.decider` reports
them as left out on purpose, with a count. That is what keeps drift in this
table visible, which the graph-API spike's blanket ignores hid.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.pipelines.architectures.qwen3_5_modulev3.weight_adapters import (
    qwen3_5_weight_rules,
)
from max.pipelines.lib import WeightRenaming, WeightRule
from transformers import PretrainedConfig

from .model_config import READOUT_WEIGHT_KEY

__all__ = ["QWEN3_5_DECIDER_WEIGHT_RULES", "qwen3_5_decider_weight_rules"]

QWEN3_5_DECIDER_WEIGHT_RULES: tuple[WeightRule, ...] = (
    # This checkpoint nests the tower directly, with no ``model.`` above it.
    WeightRenaming(r"^language_model\.", ""),
    # readout.safetensors names its one tensor "weight"; the readout module
    # names it after itself.
    WeightRenaming(r"^weight$", READOUT_WEIGHT_KEY),
)
"""The rules this checkpoint needs beyond the donor's."""


def qwen3_5_decider_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the donor's rules followed by :data:`QWEN3_5_DECIDER_WEIGHT_RULES`."""
    return (
        *qwen3_5_weight_rules(huggingface_config),
        *QWEN3_5_DECIDER_WEIGHT_RULES,
    )
