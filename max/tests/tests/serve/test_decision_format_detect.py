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

"""The decisions format is picked from the checkpoint's own files.

``joint_config.json`` outranks ``decider_config.json``, and a checkpoint
with neither keeps its chat template.
"""

from __future__ import annotations

import json
from pathlib import Path

from max.serve.router._chat_format import ChatFormat
from max.serve.router._decider_format import DeciderFormat
from max.serve.router._decision_format import detect_decision_format
from max.serve.router._joint_v4_format import (
    DECISION_CONFIG_FILE,
    JOINT_CONFIG_FILE,
    JointV4Format,
)
from tests.serve.joint_v4_fakes import _decision, _joint


def test_detect_picks_the_format_from_the_checkpoint_files(
    tmp_path: Path,
) -> None:
    assert isinstance(detect_decision_format(str(tmp_path)), ChatFormat)
    (tmp_path / "decider_config.json").write_text(
        json.dumps({"temperature": 1.0, "max_state_tokens": 1024})
    )
    assert isinstance(detect_decision_format(str(tmp_path)), DeciderFormat)
    (tmp_path / JOINT_CONFIG_FILE).write_text(json.dumps(_joint()))
    (tmp_path / DECISION_CONFIG_FILE).write_text(json.dumps(_decision()))
    assert isinstance(detect_decision_format(str(tmp_path)), JointV4Format)
