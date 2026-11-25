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

"""Refuses generation on a server whose model answers instead of generating.

Every API type is mounted whatever the checkpoint is, so a caller can post
to ``/v1/chat/completions`` on a decision model and get HTTP 200 back. For
most models that is right. For a checkpoint whose head scores answer rows
rather than a vocabulary it is not: the sampler reads ``next_token_logits``
as a distribution over token ids, so readout row indices come back as
punctuation. Nothing downstream can tell that apart from a bad prompt, and
the caller has no way to notice, so the generative routes ask the served
format whether generating means anything here
(:meth:`DecisionFormat.refuses_generation`) and answer 400 naming the
routes that do.

The format is the right thing to ask because the server chose it at startup
from the checkpoint's own protocol files, the same evidence that chose the
readout head. A checkpoint that ships no protocol keeps
:class:`~max.serve.router._chat_format.ChatFormat`, whose
:meth:`~max.serve.router._chat_format.ChatFormat.refuses_generation` is
``None``, and these routes behave exactly as they did.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from max.serve.router._decision_format import DecisionFormat

logger = logging.getLogger("max.serve")


def require_generation(request: Request, route: str) -> None:
    """Refuses a generative route the served checkpoint cannot mean.

    Args:
        request: The incoming request, for the app state.
        route: The route path, named in the log line.

    Raises:
        HTTPException: 400 carrying the format's reason, when the served
            checkpoint declares a protocol whose head scores answer rows
            instead of a vocabulary. Raised before anything is tokenized or
            queued, so a refused request costs nothing.
    """
    decision_format: DecisionFormat | None = getattr(
        request.app.state, "decision_format", None
    )
    if decision_format is None:
        # A server that never picked a format (a non-text task, or a pipeline
        # built outside the server) has nothing to say here.
        return
    reason = decision_format.refuses_generation()
    if reason is None:
        return
    logger.warning("Refusing %s: %s", route, reason)
    raise HTTPException(status_code=400, detail=reason)
