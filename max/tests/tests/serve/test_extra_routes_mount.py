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

"""Which routes an app mounts, with and without contributed ones."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.serve.api_server import ServingTokenGeneratorSettings, fastapi_app
from max.serve.config import APIType, Settings

CONTRIBUTED = "tests.serve.extra_route_fakes"
"""A route module outside MAX, exposing ``/v1/locate`` as ``router``."""


@pytest.fixture
def worker_settings() -> ServingTokenGeneratorSettings:
    """A host that has a model worker, with no pipeline behind it.

    ``memory_plan`` is ``None`` only for test servers built without one, and
    these tests build the app without entering its lifespan, so no route under
    test reaches the pipeline, the tokenizer or the worker.
    """
    return ServingTokenGeneratorSettings(
        model_factory=None,  # type: ignore[arg-type]
        pipeline_config=None,  # type: ignore[arg-type]
        tokenizer=None,  # type: ignore[arg-type]
    )


def _app(
    worker_settings: ServingTokenGeneratorSettings,
    api_types: list[APIType],
    extra_routes: list[str] | None = None,
) -> FastAPI:
    """The app ``max serve`` builds for these settings."""
    settings = Settings(
        api_types=api_types,
        extra_routes=extra_routes or [],
        use_heartbeat=False,
        disable_telemetry=True,
    )
    return fastapi_app(settings, worker_settings)


def _paths(app: FastAPI) -> set[str]:
    """The paths this app exposes.

    This FastAPI build keeps an included router as one ``_IncludedRouter``
    entry, so the mounted paths come from the generated schema.
    """
    return set(app.openapi()["paths"])


def test_empty_setting_leaves_the_stock_route_table(
    worker_settings: ServingTokenGeneratorSettings,
) -> None:
    """``MAX_SERVE_EXTRA_ROUTES`` unset mounts exactly the ``ROUTES`` table."""
    stock = _paths(_app(worker_settings, [APIType.OPENAI, APIType.SAGEMAKER]))
    assert "/v1/decisions" in stock
    assert "/v1/locate" not in stock
    assert stock == _paths(
        _app(worker_settings, [APIType.OPENAI, APIType.SAGEMAKER], [])
    )


def test_contributed_route_mounts_beside_the_openai_routes(
    worker_settings: ServingTokenGeneratorSettings,
) -> None:
    """One entry adds its own paths and changes no other path."""
    stock = _paths(_app(worker_settings, [APIType.OPENAI]))
    mounted = _paths(_app(worker_settings, [APIType.OPENAI], [CONTRIBUTED]))
    assert mounted - stock == {"/v1/locate"}
    assert "/v1/decisions" in mounted


def test_contributed_route_validates_its_own_body(
    worker_settings: ServingTokenGeneratorSettings,
) -> None:
    """A malformed locate body is refused before the engine is called.

    The engine behind ``CONTRIBUTED`` answers any state it is handed, so the
    422 can only come from ``LocateRequest`` validation.
    """
    client = TestClient(_app(worker_settings, [APIType.OPENAI], [CONTRIBUTED]))
    refused = client.post("/v1/locate", json={"state": "", "questions": {}})
    assert refused.status_code == 422, refused.text

    answered = client.post(
        "/v1/locate",
        json={"state": "billed twice", "questions": {"where": {}}},
    )
    assert answered.status_code == 200, answered.text
    assert answered.json()["answers"]["where"]["spans"][0]["text"] == (
        "billed twice"
    )


def test_sagemaker_host_mounts_no_contributed_route(
    worker_settings: ServingTokenGeneratorSettings,
) -> None:
    """A host with no OpenAI dialect has no table these routers join.

    The table it does mount is the stock one: ``/v1/decisions`` is an OpenAI
    entry, so it is missing whether or not the setting is set.
    """
    with_entry = _paths(
        _app(worker_settings, [APIType.SAGEMAKER], [CONTRIBUTED])
    )
    assert with_entry == _paths(_app(worker_settings, [APIType.SAGEMAKER]))
    assert "/v1/locate" not in with_entry
    assert "/v1/decisions" not in with_entry

    with pytest.raises(ValueError, match="no_such_routes_module_xyz"):
        _app(
            worker_settings,
            [APIType.SAGEMAKER],
            ["no_such_routes_module_xyz"],
        )


def test_bad_entry_stops_the_app_from_being_built(
    worker_settings: ServingTokenGeneratorSettings,
) -> None:
    """An entry that cannot be resolved raises before a route is mounted."""
    with pytest.raises(ValueError, match="extra_routes_import_error"):
        _app(
            worker_settings,
            [APIType.OPENAI],
            ["tests.serve.extra_routes_import_error"],
        )
