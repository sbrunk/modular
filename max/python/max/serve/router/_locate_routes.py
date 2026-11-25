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

"""``POST /v1/locate``: answer span questions over the request's state.

The route owns the wire: it validates the body, calls the engine, maps the
engine's refusals to HTTP and writes the response. Planning the sequence,
reading the state's hidden rows and turning the span head's grid into spans
belong to the engine. The server mounts one automatically when the served
checkpoint ships a span head (see
:func:`~max.serve.router._locate_flow.locate_router_for_model`); outside
packages can still contribute their own engine through
``MAX_SERVE_EXTRA_ROUTES`` (see :func:`max.serve.api_server.resolve_extra_routers`).

The engine reads hidden rows, not candidate scores. It calls ``score`` with
``label_hidden=True`` at the positions the request names, and the step returns
those rows on :attr:`TextGenerationOutput.label_hidden_rows`. No candidate
token is named, so :attr:`TextGenerationOutput.label_log_probabilities` stays
empty. The rows' provenance stays inside the engine: the response reports
spans.
"""

from __future__ import annotations

import logging
from typing import Protocol

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from max.serve.router._decisions_flow import decision_errors
from max.serve.schemas._locate import LocateRequest, LocateResponse
from pydantic import ValidationError

logger = logging.getLogger("max.serve")


class LocateEngine(Protocol):
    """Answers one locate request body with spans over its state.

    ``answer`` receives the incoming request beside the body: an engine
    that answers from the running pipeline reads it from the request's
    app state. It raises
    :class:`~max.serve.router._decisions_scoring.DecisionRequestError` for a
    body this server cannot answer and
    :class:`~max.serve.router._decisions_flow.LocateRowsUnavailable` for state
    rows it cannot read. Anything else is a server fault and becomes a logged
    500.
    """

    async def answer(
        self, locate_request: LocateRequest, request: Request
    ) -> LocateResponse: ...


def locate_router(engine: LocateEngine) -> APIRouter:
    """Builds the ``/v1/locate`` router for one engine.

    Args:
        engine: The object that plans the request and produces its spans. It is
            captured by the handler, so a server that mounts this router answers
            from it for every request.

    Returns:
        A router naming ``/v1/locate``, ready for
        ``MAX_SERVE_EXTRA_ROUTES``.
    """
    router = APIRouter(prefix="/v1")

    @router.post("/locate", response_model=None)
    async def create_locate(request: Request) -> JSONResponse:
        """Answers locate questions with spans over the request's state.

        The body is validated on the event loop and the engine awaits the
        scoring step itself; only its span head, a host graph, leaves the
        event loop.
        """
        request_id = request.state.request_id
        try:
            locate_request = LocateRequest.model_validate_json(
                await request.body()
            )
        except ValidationError as e:
            logger.warning("Validation error in request %s: %s", request_id, e)
            raise HTTPException(status_code=422, detail=str(e)) from e

        async with decision_errors(request, "/v1/locate"):
            response = await engine.answer(locate_request, request)
        return JSONResponse(content=response.model_dump(exclude_none=True))

    return router
