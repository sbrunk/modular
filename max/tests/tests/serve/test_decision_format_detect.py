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
from max.serve.router._decision_format import (
    DecisionFormat,
    detect_decision_format,
)
from max.serve.router._joint_v4_format import (
    DECISION_CONFIG_FILE,
    JOINT_CONFIG_FILE,
    JointV4Format,
)
from tests.serve.joint_v4_fakes import _decision, _joint, _settings


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


def test_a_checkpoint_with_a_vocabulary_head_may_generate(
    tmp_path: Path,
) -> None:
    """Only a readout head makes generation meaningless, so only one refusal.

    The decider format writes its option labels into the prompt and reads
    their logits out of the model's own vocabulary, so a decider checkpoint
    still has the head the chat routes sample: refusing it there would
    refuse a model that can generate.
    """
    plain = tmp_path / "plain"
    plain.mkdir()
    assert detect_decision_format(str(plain)).refuses_generation() is None
    (tmp_path / "decider_config.json").write_text(
        json.dumps({"temperature": 1.0, "max_state_tokens": 1024})
    )
    assert detect_decision_format(str(tmp_path)).refuses_generation() is None


def test_the_joint_readout_refuses_generation_with_a_pointer() -> None:
    """Protocol v4 answers from readout rows, so its reason names the count
    of those rows and the routes that read them."""
    served: DecisionFormat = JointV4Format(_settings())
    reason = served.refuses_generation()
    assert reason is not None
    assert "4 answer rows" in reason
    assert "/v1/decisions" in reason
    assert "/v1/locate" in reason
