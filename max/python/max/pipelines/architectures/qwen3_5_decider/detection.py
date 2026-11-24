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

"""The checkpoint evidence that resolves the bare ``Qwen3_5Model`` name here.

A checkpoint saved from the Qwen3.5 backbone class declares ``config.json``'s
``architectures[0]`` as the bare ``Qwen3_5Model``, with no
``ForConditionalGeneration`` or ``ForCausalLM`` head to tell MAX which
architecture wants it. Two checkpoints can share that bare name and need
different architectures, so the name alone cannot decide. What does decide is
a file only the released decision checkpoint carries: ``decision_config.json``,
the answer-code table the vendor's protocol, readout and calibration all read.

The predicate runs while the declared name is being resolved, before this
architecture's module is imported, so it stays a directory check and reaches
nothing inside the checkpoint. The decision head itself needs no test here:
:class:`.model_config.Qwen3_5DeciderConfig` counts its rows from
``readout.weight`` at finalize, and a checkpoint missing that tensor stops
there with the head named.
"""

from __future__ import annotations

from pathlib import Path

DECISION_CONFIG_FILE = "decision_config.json"
"""The released decision checkpoint's answer-code table.

Its presence, not its contents, is the alias evidence: the file's absence
means the checkpoint is not a decision checkpoint, and a malformed file fails
later in the protocol layer, where the failure names it."""


def accepts_decider_checkpoint(model_path: str | Path) -> bool:
    """Whether the checkpoint at ``model_path`` is a released decision one.

    Args:
        model_path: The resolved model directory, as the models config holds
            it (a local path; a Hub id resolves to a cache directory before
            this runs).

    Returns:
        True when the directory carries :data:`DECISION_CONFIG_FILE`.
    """
    return (Path(model_path) / DECISION_CONFIG_FILE).is_file()
