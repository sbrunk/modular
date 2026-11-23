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

"""A contributed route module that fails while being imported.

``resolve_extra_routers`` catches anything raised here and re-raises a
``ValueError`` naming the setting entry, the way ``import_custom_architectures``
reports a bad ``--custom-architectures`` spec.
"""

raise RuntimeError("this route module needs a config file that is not there")
