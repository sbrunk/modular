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

"""Checks how the decider registers, without loading a checkpoint.

The registration carries the correctness rules the serving path depends on:
its own name (the checkpoint's declared name reaches it through the alias
table, not by claiming the bare name here), the execution pins applied by
``required_arguments``, and the batch processor that computes the segment
edges the state rewind reads.
"""

from __future__ import annotations

from max.pipelines.architectures.qwen3_5_decider import (
    qwen3_5_decider_arch,
)
from max.pipelines.architectures.qwen3_5_decider.batch_processor import (
    Qwen3_5DeciderBatchProcessor,
)
from max.pipelines.architectures.qwen3_5_decider.decider import (
    Qwen3_5DeciderDecoder,
)
from max.pipelines.lib import GraphRole
from max.pipelines.lib.arch_lookup import find_architecture
from max.pipelines.modeling.types import PipelineTask


def test_decider_resolves_under_its_own_name() -> None:
    """The built-in answers to ``Qwen3_5DeciderModel``, suffixed or not.

    ``prefer_module_v3`` looks the ``_ModuleV3`` name up first and falls back
    to the plain name when only one variant is registered; the decider builds
    only the ModuleV3 graph, so it registers once and reaches both lookups
    through that fallback.
    """
    plain = find_architecture("Qwen3_5DeciderModel")
    modulev3 = find_architecture("Qwen3_5DeciderModel", prefer_module_v3=True)
    assert plain is qwen3_5_decider_arch
    assert modulev3 is qwen3_5_decider_arch


def test_bare_backbone_name_is_not_registered() -> None:
    """Nothing answers the bare ``Qwen3_5Model`` name by registration.

    That name resolves only through the alias table, so the lookup without a
    checkpoint to test finds no architecture rather than guessing.
    """
    assert find_architecture("Qwen3_5Model") is None


def test_registration_pins_whole_row_execution() -> None:
    """The pins that keep a bidirectional state correct are declared."""
    assert qwen3_5_decider_arch.task is PipelineTask.TEXT_GENERATION
    assert qwen3_5_decider_arch.required_arguments == {
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "state_pool_dtype": "float32",
        "max_batch_size": 1,
    }
    # A wider batch is not merely slower here: the segment table names spans
    # inside one row, so a multi-row step rewinds every row at the first
    # row's edges. The batch processor refuses such a step outright.
    assert qwen3_5_decider_arch.required_arguments["max_batch_size"] == 1
    assert qwen3_5_decider_arch.checkpoints_recurrent_state
    assert qwen3_5_decider_arch.batching is Qwen3_5DeciderBatchProcessor
    assert qwen3_5_decider_arch.modules and (
        qwen3_5_decider_arch.modules.get(GraphRole.DECODER)
        is Qwen3_5DeciderDecoder
    )
