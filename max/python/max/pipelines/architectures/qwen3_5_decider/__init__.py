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

"""A MAX ModuleV3 architecture that runs the pplx decider checkpoint.

The checkpoint is a Qwen3.5 hybrid (gated full attention every fourth layer,
gated DeltaNet elsewhere) whose ``lm_head`` is tied away and whose decision
head is the single 255-row tensor in ``readout.safetensors``. It registers as
``Qwen3_5DeciderModel``: a text-generation decoder whose
``next_token_logits`` are readout scores, one row per protocol-v4 question,
taken at that question's answer marker. The checkpoint declares only the bare
backbone name in its ``config.json``, and :mod:`.detection` is what resolves
that name here.
"""

from __future__ import annotations

from .arch import qwen3_5_decider_arch

ARCHITECTURES = [qwen3_5_decider_arch]

__all__ = ["ARCHITECTURES", "qwen3_5_decider_arch"]
