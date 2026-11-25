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

"""Registers the decider on ModuleV3 as its own architecture.

``name`` is ``Qwen3_5DeciderModel``, MAX's own. The checkpoint's
``config.json`` declares ``architectures[0] == "Qwen3_5Model"``: the bare
backbone class name, not a ``ForConditionalGeneration`` or ``ForCausalLM``
head, and no other built-in claims that bare name. A checkpoint declaring it
is routed to this package by the name alias in
:mod:`max.pipelines.architectures`, which
consults :func:`.detection.accepts_decider_checkpoint` first, so
``max serve --model-path <checkpoint>`` resolves with no flag and no
``--custom-architectures``.

The registration follows
``max.pipelines/architectures/qwen3_5_modulev3/arch.py``: a ``DECODER`` role
module, the donor's tokenizer, context type and config base,
``checkpoints_recurrent_state=True`` so the gated-DeltaNet state leaves in the
cache tree select the Jenga pool, and the donor's weight rules with this
checkpoint's two extras. The batch processor is the donor's subclass in
:mod:`~.batch_processor`, which hands the rewound DeltaNet its segment edges.

Deltas from the donor registration:

- **No ``VISION`` module, no M-RoPE, no speculative pairing, no format/tool/
  reasoning parser.** Protocol v4 questions are text, and the graph is scored
  once per row rather than generated from.
- **Single device.** ``multi_gpu_supported`` stays False: this architecture
  is exercised on one accelerator, and the tensor-parallel paths it inherits
  are untested with the readout head and the state rewind.
- **The execution rules are asserted, not assumed.** A bidirectional state is
  correct only when it is seen once, whole, so the registration itself pins
  ``enable_chunked_prefill=False`` (a question must not split across steps),
  ``enable_prefix_caching=False`` (a cache hit would feed a row's uncached
  suffix and resume a state that consumed a different prefix), and
  ``state_pool_dtype="float32"`` (the reference runs the gated-DeltaNet
  recurrence in float32, and the pool dtype is what the state kernels round
  to). ``PipelineConfig`` applies these over the caller's values with a warning
  (:func:`max.pipelines.lib.config.config._apply_required_arguments`), so
  ``max serve --model-path <checkpoint>`` gets them without a flag. Nothing
  else in MAX's config resolution writes either one back on. ``--force`` is
  the one way past them,
  and :func:`qwen3_5_decider.model_config.require_whole_row_execution`
  closes that route: it refuses the build and names the flags before a graph
  can serve probabilities computed against another question's state.
"""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.qwen3_5_modulev3.context import (
    Qwen3_5TextAndVisionContext,
)
from max.pipelines.architectures.qwen3_5_modulev3.tokenizer import (
    Qwen3_5Tokenizer,
)
from max.pipelines.lib import GraphRole, SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Qwen3_5DeciderBatchProcessor
from .decider import Qwen3_5DeciderDecoder
from .model_config import Qwen3_5DeciderConfig
from .weight_adapters import qwen3_5_decider_weight_rules

__all__ = ["qwen3_5_decider_arch"]

qwen3_5_decider_arch = SupportedArchitecture(
    name="Qwen3_5DeciderModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["perplexity-ai/pplx-decider-flash-v1.3"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen3_5DeciderConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen3_5DeciderConfig.SUPPORTED_ENCODINGS,
    modules={GraphRole.DECODER: Qwen3_5DeciderDecoder},
    tokenizer=Qwen3_5Tokenizer,
    context_type=Qwen3_5TextAndVisionContext,
    input_modalities={InputModality.TEXT},
    weight_rules=qwen3_5_decider_weight_rules,
    use_weight_rules=True,
    # The recurrent-state leaves are what make the cache tree hold the
    # gated-DeltaNet pools; without this the hybrid tower has nowhere to keep
    # its state.
    checkpoints_recurrent_state=True,
    config=Qwen3_5DeciderConfig,
    batching=Qwen3_5DeciderBatchProcessor,
    # Device graph capture targets repeated decode steps; this graph runs one
    # prefill per request.
    supports_device_graph_capture=False,
    # The plain text-generation pipeline owns the scoring path the decisions
    # routes use; the overlap scheduler would add a second execution path.
    supports_overlap_scheduler=False,
    required_arguments={
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "state_pool_dtype": "float32",
        # The segment table the rewound DeltaNet traces names spans within one
        # row and no row axis, so a step batching several rows would rewind
        # every row at the first row's edges. One request per forward pass
        # until the table carries a row axis.
        "max_batch_size": 1,
    },
)
