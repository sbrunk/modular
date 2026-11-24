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

"""The decider's MLP with each of its three projections called as a module.

``max.nn.common_layers.quant_linear.QuantizedMLP.forward`` fuses ``gate_proj``
and ``up_proj`` into one matmul through ``quant_ops.concat_weights`` and calls
only ``down_proj`` as a module. A fold that hangs a site on a projection's
``forward`` (:func:`.adapter._delta_class`) then runs the delta at
``down_proj`` and skips the two wider sites: 48.23 M of this adapter's 72.35 M
MLP elements would be bound to weights no op consumes.

``LocateMLP`` keeps the donor's parameters and calls the three projections as
modules, so all three wrapped forwards run and each delta multiplies the same
input its base matmul got, as ``LoRALinear.forward`` arranges
(``locate.py:83-92``). The fusion is given up for a folded build: two matmuls
of ``[intermediate, hidden]`` against one of ``[2 * intermediate, hidden]``.
Only a folded build takes this class, so the decisions path keeps the donor's
fused body and the probabilities it has always answered.
"""

from __future__ import annotations

from max.nn.common_layers.quant_linear import QuantizedMLP
from max.tensor import Tensor

__all__ = ["LocateMLP"]


class LocateMLP(QuantizedMLP):
    """The donor's gated MLP, with ``gate_proj``/``up_proj`` called as modules.

    Construction, parameters and the quant config stay the donor's: the fold
    re-classes the built instance in place, so the weight paths
    ``mlp.gate_proj.weight`` and its scales are the ones the checkpoint names.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``down_proj(act(gate_proj(x)) * up_proj(x))``.

        Args:
            x: The block's normed hidden states, ``[rows, hidden_size]``.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.activation_function(gate) * up)
