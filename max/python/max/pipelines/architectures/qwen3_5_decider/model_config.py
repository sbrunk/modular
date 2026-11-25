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

"""The decider's config: the Qwen3.5 hybrid geometry, text-only, with a head.

Derived from
:class:`max.pipelines.architectures.qwen3_5_modulev3.model_config.Qwen3_5Config`,
which already reads this checkpoint's ``text_config``: the 32-layer hybrid
schedule, the gated-DeltaNet geometry, the partial-RoPE factor, the
gated-DeltaNet scalar dtypes, and the cache tree
(:meth:`Qwen3_5Config.construct_kv_params`) whose recurrent leaves are what
select the Jenga state pool.

Deltas the decider's config layers on top of the donor's:

- **No vision tower, no M-RoPE.** The checkpoint ships a ``vision_config`` and
  an ``mrope_section``, so the donor config would build a tower and take
  per-token 3-axis positions. Protocol v4 is text here, and a text-only row
  has one position axis: :meth:`initialize_from_config` clears both, which
  also puts the rotary table back on the static, per-row-index form
  (:attr:`Qwen3_5Config.mrope_enabled`).
- **The head is the 255-row readout**, counted in :meth:`finalize` from
  ``readout.weight``. ``lm_head`` stays absent, as the donor leaves it for a
  ``tie_word_embeddings`` checkpoint.
- **The state pools are float32**, which the registration pins through
  ``required_arguments``; :attr:`state_dtype` then reports it for both the
  cache leaves and the graph inputs.
- :meth:`initialize_from_config` refuses a config that still asks for prefix
  reuse, chunked prefill, or state pools below float32
  (:func:`require_whole_row_execution`), so the only way to serve this
  architecture is one whole row per step over a float32 pool.
- :meth:`get_vision_cache_row_spec` and
  :meth:`estimate_vision_cache_entry_bytes` report no tower, so
  :func:`max.pipelines.lib.interfaces.arch_config.arch_has_vision_tower` says
  False for a checkpoint whose ``config.json`` says otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.qwen3_5_modulev3.model_config import (
    Qwen3_5Config,
)
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self

__all__ = [
    "EMBED_LOCATE_KEY",
    "LOCATE_MARKER_ID",
    "READOUT_WEIGHT_KEY",
    "Qwen3_5DeciderConfig",
    "require_whole_row_execution",
]

READOUT_WEIGHT_KEY = "readout.weight"
"""The adapted checkpoint name of the decision head's single tensor.

``readout.safetensors`` stores it as ``weight``; the decider's weight rules
name it after the module that reads it (:mod:`.weight_adapters`).
"""

EMBED_LOCATE_KEY = "embed_tokens.locate"
r"""The adapted name of the trained ``<|locate|>`` embedding row, which the
checkpoint stores as ``language_model.embed_tokens.locate`` and the
``^language_model\.`` renaming brings to this path."""

LOCATE_MARKER_ID = 248077
"""The ``<|locate|>`` token id: ``locate_config.json``'s ``marker_id``, and the
token that turns the adapter on for the row it sits in.

The graph compares it against the ragged tokens to build the gate, so a
checkpoint trained on a different marker needs this constant changed with the
adapter; no build-time read of ``tokenizer.json`` covers it.
"""


def require_whole_row_execution(pipeline_config: PipelineConfig) -> None:
    """Refuses a launch that splits a row, resumes a state, or thins the pools.

    The decider's state is bidirectional: a gated-DeltaNet memory and a set of
    attention pages that depend on tokens *after* the position they were
    written at, over a context that carries several questions sharing one
    state. A row is therefore correct only when one step sees it whole and
    nothing is reused from a step that saw different tokens. The recorded
    numbers add a third condition: the gated-DeltaNet pools hold float32. The
    registration pins all three with ``required_arguments``
    (:mod:`qwen3_5_decider.arch`):
    ``enable_prefix_caching=False``, ``enable_chunked_prefill=False`` and
    ``state_pool_dtype="float32"``.
    :func:`max.pipelines.lib.config.config._apply_required_arguments` applies
    them over the caller's values on every launch route, and skips them when
    ``runtime.force`` is set. That is the one way a resolved config can still
    ask for reuse, or leave the pool dtype at the default, and this check closes
    the route at pipeline-build time, before a request is served. Each refusal
    names the setting and what it changes.

    Args:
        pipeline_config: The resolved config, after the registration's required
            arguments have landed.

    Raises:
        ValueError: If prefix caching or chunked prefill is on, or if
            ``state_pool_dtype`` is not ``"float32"``. MAX serves a prefix hit by
            moving the row's processing window forward
            (``TokenState.skip_processing``), and chunked prefill splits a row
            against ``max_batch_input_tokens``; either way the graph is fed a
            suffix rather than the row. With the pool dtype unset or lowered,
            :attr:`Qwen3_5Config.state_dtype` answers ``compute_dtype`` and the
            state kernels round the conv window and the recurrent state to that.
    """
    kv_cache = pipeline_config.model.kv_cache
    runtime = pipeline_config.runtime
    still_on: list[str] = []
    if kv_cache.enable_prefix_caching:
        still_on.append(
            "model.kv_cache.enable_prefix_caching=True (--enable-prefix-caching):"
            " a cache hit resumes the state a previous row left, and a"
            " bidirectional state left by one question is not the state this"
            " question's tokens produce"
        )
    if runtime.enable_chunked_prefill:
        still_on.append(
            "runtime.enable_chunked_prefill=True (--enable-chunked-prefill):"
            " a split row is encoded over several steps, so the second half"
            " attends to keys and a state from a step that never saw the whole"
            " row"
        )
    if kv_cache.state_pool_dtype != "float32":
        still_on.append(
            "model.kv_cache.state_pool_dtype="
            f"{kv_cache.state_pool_dtype!r} (--state-pool-dtype float32): the"
            " state kernels round the conv window and the recurrent state to the"
            " pool dtype, and Qwen3_5Config.state_dtype answers compute_dtype"
            " when the field is unset, so a forced launch runs bfloat16"
            " gated-DeltaNet pools while every recorded decision and Locate"
            " number comes from float32"
        )
    if still_on:
        raise ValueError(
            "Qwen3_5DeciderModel cannot be served with this"
            " configuration: every protocol-v4 row must be fed whole, once,"
            " over state pools holding the precision the recorded numbers come"
            " from. This architecture pins enable_prefix_caching=False,"
            " enable_chunked_prefill=False and"
            ' state_pool_dtype="float32" through SupportedArchitecture'
            ".required_arguments, so a resolved config with any one of them"
            " changed means runtime.force (--force) skipped that pin:"
            + "".join(f"\n  - {item}" for item in still_on)
            + "\nDrop --force, or pass --no-enable-prefix-caching,"
            " --no-enable-chunked-prefill and --state-pool-dtype float32"
            " explicitly; the alternative is option probabilities scored"
            " against another question's state, or a state rounded to the"
            " compute dtype."
        )


def _read_adapter(state_dict: Mapping[str, WeightData]) -> tuple[int, int]:
    """Returns ``(rank, marker_id)`` for the Locate adapter in the checkpoint.

    ``(0, LOCATE_MARKER_ID)`` when no ``*.lora_a`` tensor is present: that is
    the weight set of a plain backbone, the decoder builds no adapter sites, and
    the audit excuses nothing in the adapter group.

    Args:
        state_dict: The adapted checkpoint's tensors, after the weight rules
            renamed them.

    Raises:
        ValueError: If the ``lora_a`` tensors hold two ranks, if a projection
            carries one half of a pair, or if the ``embed_tokens.locate`` row is
            missing while projections carry a delta.
    """
    ranks: set[int] = set()
    pair_sites: set[tuple[str, str]] = set()
    for name, weight in state_dict.items():
        site, _, part = name.rpartition(".")
        if part in ("lora_a", "lora_b"):
            pair_sites.add((site, part))
            if part == "lora_a":
                ranks.add(int(weight.shape[0]))
    if not pair_sites:
        return 0, LOCATE_MARKER_ID
    if len(ranks) != 1:
        raise ValueError(
            f"the Locate adapter declares {len(ranks)} ranks {sorted(ranks)};"
            " one compiled graph builds one rank of delta"
        )
    sites = {site for site, _ in pair_sites}
    unpaired = sorted(
        sites - {s for s, p in pair_sites if p == "lora_b"}
    ) + sorted(sites - {s for s, p in pair_sites if p == "lora_a"})
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} Locate adapter site(s) hold a lora_a or a lora_b"
            f" and not both: {unpaired[:5]}"
        )
    if EMBED_LOCATE_KEY not in state_dict:
        raise ValueError(
            f"the adapter writes {len(sites)} projection deltas and the"
            f" checkpoint has no {EMBED_LOCATE_KEY!r} row; the reference"
            " enables the marker embedding and the LoRA in one pass"
        )
    return ranks.pop(), LOCATE_MARKER_ID


@dataclass(kw_only=True)
class Qwen3_5DeciderConfig(Qwen3_5Config):
    """The Qwen3.5 hybrid config as the decider graph reads it."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    """The serving dtype. The fp32 copy of the weights is a measurement
    substrate; the decider serves bfloat16."""

    readout_rows: int = 0
    """Rows of the readout: the answer codes the head scores (255 here)."""

    adapter_rank: int = 0
    """Rank of the Locate LoRA in the adapted checkpoint, 64 for the shipped
    adapter, and 0 when the checkpoint holds no ``*.lora_a`` tensor. The text
    model builds one ``(lora_a, lora_b)`` pair per projection at this rank."""

    locate_marker_id: int = LOCATE_MARKER_ID
    """The token id that gates the adapter. Read with the rank in
    :meth:`finalize`, and irrelevant when no adapter is present."""

    checkpoint_tensor_names: frozenset[str] = frozenset()
    """Every tensor name the adapted checkpoint holds, recorded for the
    weight-coverage audit the decoder runs when it builds."""

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Reads the donor's config, then removes the vision and M-RoPE parts.

        ``vision_config`` is what the donor uses as "an image can appear in
        this context", so clearing it drops the tower the decider never
        registers and turns M-RoPE off; ``mrope_section`` is cleared with it
        so the static rotary table is the one the graph builds.

        :meth:`initialize_from_config` also refuses an impossible launch: the
        hook is reached from ``ArchConfig.initialize`` while the pipeline is
        being built, and it receives the resolved config, so
        :func:`require_whole_row_execution` can see whether the registration's
        pins are still in force.

        Raises:
            ValueError: If prefix caching or chunked prefill survived config
                resolution, or if the state pool dtype is not float32. See
                :func:`require_whole_row_execution`.
        """
        require_whole_row_execution(pipeline_config)
        config = super().initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )
        config.vision_config = None
        config.mrope_section = None
        config.mrope_without_encoder = False
        return config

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Records the readout width, the adapter rank, and the checkpoint's names.

        Raises:
            ValueError: If the adapted checkpoint has no readout tensor. The
                decision head comes from ``readout.safetensors``, whose single
                tensor the weight rules name :data:`READOUT_WEIGHT_KEY`; a
                checkpoint without it cannot be scored.
            ValueError: If the Locate adapter states two ranks, leaves a
                projection half a pair, or writes projection deltas with no
                marker-embedding row. See :func:`_read_adapter`.
        """
        config = super().finalize(huggingface_config, state_dict, outputs)
        readout = state_dict.get(READOUT_WEIGHT_KEY)
        if readout is None:
            raise ValueError(
                f"the adapted checkpoint has no {READOUT_WEIGHT_KEY!r};"
                " the decider's head comes from readout.safetensors, whose"
                " single tensor 'weight' the decider's weight rules rename"
            )
        config.readout_rows = int(readout.shape[0])
        config.adapter_rank, config.locate_marker_id = _read_adapter(state_dict)
        config.checkpoint_tensor_names = frozenset(state_dict)
        return config

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Zero: the decider compiles no vision encoder for any checkpoint."""
        del huggingface_config
        return 0

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """``None``: no tower, so no vision-encoder cache rows to reserve."""
        del huggingface_config
        return None
