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

"""Where the label channel lands, with the channel off and on.

``_grouped_outputs`` names a decoder's outputs by group, and a device graph
capture hands its outputs back as a plain list, so the positional mappers have to
agree with the leaf order. ``ModelOutputs`` puts ``label_logits`` last among the
tensor outputs, and ``bind_captured_outputs`` refuses a capture whose leaf count
it cannot name, since a second captured buffer is hidden states or label rows
and the two bind to different fields. The graph-API half of the seam,
``logits_outputs``, ``decoder_outputs`` and ``_graph_outputs``, names the channel
too, where it used to stop at ``logit_offsets``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.nn.legacy.transformer import logits_outputs
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.interfaces import (
    ModelOutputs,
    bind_captured_outputs,
    process_ragged_kv_outputs,
)
from max.pipelines.lib.module_registration import (
    _grouped_outputs,
    _role_outputs,
)
from max.pipelines.lib.registration_graph_api import (
    _graph_outputs,
    decoder_outputs,
)
from max.pipelines.lib.role_contracts import DecoderOutputs

FIELDS = (
    "logits",
    "next_token_logits",
    "logit_offsets",
    "hidden_states",
    "label_logits",
)
"""The :class:`ModelOutputs` fields these cases read back."""


def _named(name: str) -> Any:
    """A stand-in output buffer named for the field it belongs to."""
    return name


def _fields(outputs: ModelOutputs) -> dict[str, str]:
    """Which buffer landed in which field."""
    return {
        name: getattr(outputs, name)
        for name in FIELDS
        if getattr(outputs, name) is not None
    }


def _grouped(with_channel: bool, with_hidden: bool) -> Any:
    """The grouped outputs of one build, as the registration writes them."""
    return _grouped_outputs(
        DecoderOutputs(
            next_token_logits=_named("next_token_logits"),
            label_logits=_named("label_logits") if with_channel else None,
            hidden_states=_named("hidden_states") if with_hidden else None,
        ),
        ReturnHiddenStates.LAST if with_hidden else ReturnHiddenStates.NONE,
    )


def test_leaf_order_and_role_read_back() -> None:
    """The channel is the leaf a build adds, and the role names it back."""
    for with_channel, with_hidden in (
        (False, False),
        (True, False),
        (True, True),
    ):
        grouped = _grouped(with_channel, with_hidden)
        leaves = list(grouped["logits"])
        if "hidden_states" in grouped:
            leaves += [
                f"hidden_states/{key}" for key in grouped["hidden_states"]
            ]
        back = _role_outputs(grouped)
        landed = {
            name: getattr(back, name)
            for name in ("next_token_logits", "label_logits", "hidden_states")
            if getattr(back, name) is not None
        }
        print(
            f"channel={with_channel} hidden={with_hidden}: leaves {leaves}; "
            f"role fields {landed}"
        )
        assert (back.label_logits is not None) is with_channel
        assert (back.hidden_states is not None) is with_hidden
        group = list(grouped["logits"])
        assert (group[-1] == "label_logits") is with_channel


def test_a_capture_without_the_channel_binds_the_next_token_rows() -> None:
    """One leaf, and the field it has always landed in stays the same."""
    outputs = bind_captured_outputs([_named("next_token_logits")])
    landed = _fields(outputs)
    print(f"capture without the channel: {landed}")
    assert landed == {"logits": "next_token_logits"}
    assert isinstance(outputs, ModelOutputs)


def test_a_capture_with_the_channel_is_refused_by_name() -> None:
    """Two leaves are hidden states or label rows, and a capture names neither."""
    with pytest.raises(ValueError, match="returned 2 outputs"):
        bind_captured_outputs(
            [_named("next_token_logits"), _named("label_logits")]
        )


def _buffer(seed: int) -> Buffer:
    """A real driver buffer, which the legacy mapper asserts on."""
    return Buffer.from_numpy(np.arange(2, dtype=np.int64) + seed).to(CPU())


def test_the_legacy_ragged_mapper_refuses_a_fifth_buffer() -> None:
    """The ragged+KV mapper names four buffers at most, and says so."""
    with pytest.raises(ValueError, match="got 5"):
        process_ragged_kv_outputs(
            [_buffer(0), _buffer(10), _buffer(20), _buffer(30), _buffer(40)],
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.LAST,
        )


def test_the_legacy_ragged_mapper_keeps_the_shapes_it_names() -> None:
    """Four named buffers still land in the fields they always did."""
    outputs = process_ragged_kv_outputs(
        [_buffer(0), _buffer(10), _buffer(20), _buffer(30)],
        return_logits=ReturnLogits.VARIABLE,
        return_hidden_states=ReturnHiddenStates.LAST,
    )
    print(
        "legacy ragged mapper: "
        f"{[name for name in FIELDS if getattr(outputs, name) is not None]}"
    )
    assert outputs.logit_offsets is not None
    assert outputs.hidden_states is not None
    assert outputs.label_logits is None


def test_model_outputs_declares_the_channel_after_the_hidden_states() -> None:
    """The dataclass order matches the leaf order a graph returns."""
    names = [field.name for field in fields(ModelOutputs)]
    print(f"ModelOutputs fields: {names}")
    assert names.index("label_logits") > names.index("hidden_states")


def _role(with_channel: bool) -> Any:
    """The role's outputs of one build, with named stand-in buffers."""
    return DecoderOutputs(
        next_token_logits=_named("next_token_logits"),
        label_logits=_named("label_logits") if with_channel else None,
    )


def test_logits_outputs_names_the_channel_last_in_the_group() -> None:
    """A graph-API model's group keeps the registration's leaf order."""
    for with_channel in (False, True):
        grouped = logits_outputs(
            _named("next_token_logits"),
            None,
            None,
            ReturnHiddenStates.NONE,
            (),
            _role(with_channel).label_logits,
        )
        leaves = list(grouped["logits"])
        print(f"logits_outputs channel={with_channel}: leaves {leaves}")
        assert (
            leaves == ["next_token_logits", "label_logits"]
            if with_channel
            else ["next_token_logits"]
        )


def test_decoder_outputs_reads_the_channel_back() -> None:
    """A graph-API decoder that set the field keeps it on the way out."""
    for with_channel in (False, True):
        grouped = logits_outputs(
            _named("next_token_logits"),
            None,
            None,
            ReturnHiddenStates.NONE,
            (),
            _role(with_channel).label_logits,
        )
        back = decoder_outputs(grouped)
        landed = {
            name: getattr(back, name)
            for name in ("next_token_logits", "label_logits")
            if getattr(back, name) is not None
        }
        print(f"decoder_outputs channel={with_channel}: fields {landed}")
        assert (back.label_logits == "label_logits") is with_channel


def test_graph_outputs_export_the_channel() -> None:
    """The exported graph's leaf list grows by the channel, and keeps order."""
    for with_channel in (False, True):
        grouped: Any = _graph_outputs(
            _role(with_channel), ReturnHiddenStates.NONE
        )
        leaves = list(grouped["logits"])
        print(f"_graph_outputs channel={with_channel}: leaves {leaves}")
        assert leaves == (
            ["next_token_logits", "label_logits"]
            if with_channel
            else ["next_token_logits"]
        )
