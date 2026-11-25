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

"""The bare ``Qwen3_5Model`` name resolves to the decider only for decider
checkpoints.

A checkpoint saved from the Qwen3.5 backbone class declares the bare
backbone name in ``config.json``. The alias table reroutes that name to
``Qwen3_5DeciderModel`` when the checkpoint directory carries
``decision_config.json``, and leaves it alone otherwise: the whole point of
the predicate is that a name two kinds of checkpoint can share decides by the
checkpoint's files, not by whichever architecture was registered first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from max.pipelines.architectures.qwen3_5_decider.detection import (
    DECISION_CONFIG_FILE,
    accepts_decider_checkpoint,
)
from max.pipelines.lib._model_components import architecture_name_for


class _DeclaredModel:
    """The two fields :func:`architecture_name_for` reads off a model config."""

    def __init__(self, architecture: str, model_path: str) -> None:
        self.architecture_name = architecture
        self.model_path = model_path


def _models(architecture: str, model_path: str) -> dict[str, Any]:
    return {"main": cast(Any, _DeclaredModel(architecture, model_path))}


def test_marker_file_names_the_decider(tmp_path: Path) -> None:
    """A checkpoint directory holding the decision table is a decider."""
    (tmp_path / DECISION_CONFIG_FILE).write_text("{}")
    assert accepts_decider_checkpoint(tmp_path)
    resolved = architecture_name_for(_models("Qwen3_5Model", str(tmp_path)), {})
    assert resolved == "Qwen3_5DeciderModel"


def test_bare_name_without_the_marker_stands(tmp_path: Path) -> None:
    """The same declared name over a plain checkpoint directory is untouched.

    This is the capture the alias exists to avoid: rerouting by name alone
    would hand an ordinary backbone checkpoint to the readout architecture
    and answer with decision scores nobody asked for.
    """
    assert not accepts_decider_checkpoint(tmp_path)
    resolved = architecture_name_for(_models("Qwen3_5Model", str(tmp_path)), {})
    assert resolved == "Qwen3_5Model"


def test_alias_does_not_touch_other_names(tmp_path: Path) -> None:
    """A marker file cannot re-title a named-head checkpoint."""
    (tmp_path / DECISION_CONFIG_FILE).write_text("{}")
    resolved = architecture_name_for(
        _models("Qwen3_5ForCausalLM", str(tmp_path)), {}
    )
    assert resolved == "Qwen3_5ForCausalLM"
