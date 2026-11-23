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

"""Tests for the ``MAX_SERVE_EXTRA_ROUTES`` setting."""

from __future__ import annotations

import re

import pytest
from fastapi.routing import APIRoute
from max.serve.api_server import resolve_extra_routers
from max.serve.config import Settings


def test_extra_routes_default_is_empty() -> None:
    """An unset setting contributes no routers."""
    assert Settings().extra_routes == []


def test_extra_routes_env_names_module_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alias takes a JSON list: one module path per contributed route."""
    monkeypatch.setenv(
        "MAX_SERVE_EXTRA_ROUTES", '["mypkg.routes", "otherpkg.routes"]'
    )
    assert Settings().extra_routes == ["mypkg.routes", "otherpkg.routes"]


@pytest.mark.parametrize(
    "entry",
    [
        # No such module.
        "no_such_routes_module_xyz",
        # A module that raises while being imported.
        "tests.serve.extra_routes_import_error",
        # A module that imports and exposes no router.
        "max.serve.config",
    ],
)
def test_extra_routes_bad_entry_names_it(entry: str) -> None:
    """Every entry this setting cannot use is refused by its own text.

    The import failure a module raises escapes with a traceback naming no
    entry, so the resolver re-raises and puts the entry in the message.
    """
    with pytest.raises(ValueError, match=re.escape(entry)):
        resolve_extra_routers(Settings(extra_routes=[entry]))


def test_extra_routes_resolves_a_module_exposing_router() -> None:
    """A module path whose ``router`` attribute is an ``APIRouter`` resolves."""
    routers = resolve_extra_routers(
        Settings(extra_routes=["tests.serve.extra_route_fakes"])
    )
    # ``router.routes`` is a list of ``BaseRoute``, and only ``APIRoute``
    # carries the path the mount has to answer on.
    paths = [
        route.path
        for router in routers
        for route in router.routes
        if isinstance(route, APIRoute)
    ]
    assert paths == ["/v1/locate"]
