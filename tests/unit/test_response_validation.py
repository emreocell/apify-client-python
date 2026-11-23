from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, EmailStr, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from apify_client._logging import _injects_client_details_to_log_context, log_context
from apify_client._models import UserPublicInfo
from apify_client._utils.http import validate_response
from apify_client.errors import ApifyClientError, ResponseValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_httpserver import HTTPServer

    from apify_client import ApifyClient, ApifyClientAsync

pytestmark = pytest.mark.usefixtures('http_client_classes')

# A body with the response envelope whose `data` fits none of the models. The terminal `status` lets `wait_for_finish`
# stop polling.
_MISMATCHED_BODY = {'data': {'status': 'SUCCEEDED', 'unexpected': 'shape'}}

_CASES = [
    pytest.param('/v2/datasets/ds-1', 'GET', lambda c: c.dataset('ds-1').get(), id='single_get'),
    pytest.param('/v2/datasets', 'GET', lambda c: c.datasets().list(), id='collection_list'),
    pytest.param('/v2/datasets', 'POST', lambda c: c.datasets().get_or_create(), id='collection_get_or_create'),
    pytest.param('/v2/datasets/ds-1', 'PUT', lambda c: c.dataset('ds-1').update(name='x'), id='single_update'),
    pytest.param('/v2/actor-runs/run-1', 'GET', lambda c: c.run('run-1').wait_for_finish(), id='wait_for_finish'),
]


def _assert_response_validation_error(error: ResponseValidationError, *, path: str, method: str) -> None:
    assert isinstance(error, ApifyClientError)
    assert isinstance(error, ValidationError)
    assert isinstance(error, ValueError)
    assert error.http_method == method
    assert error.url is not None
    assert error.url.endswith(path)
    assert error.error_count() > 0
    assert error.errors()
    assert isinstance(error.__cause__, ValidationError)
    assert f'{method} {error.url}' in str(error)


@pytest.mark.parametrize(('path', 'method', 'call'), _CASES)
def test_mismatched_response_raises_response_validation_error(
    httpserver: HTTPServer,
    sync_client: ApifyClient,
    path: str,
    method: str,
    call: Callable[[ApifyClient], Any],
) -> None:
    """A response that does not match its model raises `ResponseValidationError` with the request details."""
    httpserver.expect_request(path, method=method).respond_with_json(_MISMATCHED_BODY)

    with pytest.raises(ResponseValidationError) as exc_info:
        call(sync_client)

    _assert_response_validation_error(exc_info.value, path=path, method=method)


@pytest.mark.parametrize(('path', 'method', 'call'), _CASES)
async def test_mismatched_response_raises_response_validation_error_async(
    httpserver: HTTPServer,
    async_client: ApifyClientAsync,
    path: str,
    method: str,
    call: Callable[[ApifyClientAsync], Any],
) -> None:
    """A response that does not match its model raises `ResponseValidationError` with the request details."""
    httpserver.expect_request(path, method=method).respond_with_json(_MISMATCHED_BODY)

    with pytest.raises(ResponseValidationError) as exc_info:
        await call(async_client)

    _assert_response_validation_error(exc_info.value, path=path, method=method)


def test_response_validation_error_url_has_no_query_string(httpserver: HTTPServer, sync_client: ApifyClient) -> None:
    """The URL on the error leaves out the query parameters of the request."""
    httpserver.expect_request('/v2/datasets', method='GET').respond_with_json(_MISMATCHED_BODY)

    with pytest.raises(ResponseValidationError) as exc_info:
        sync_client.datasets().list(limit=1, desc=True)

    assert exc_info.value.url is not None
    assert '?' not in exc_info.value.url


def test_user_get_falls_back_to_public_info(httpserver: HTTPServer, sync_client: ApifyClient) -> None:
    """A user body that fits only the public model still returns the public info."""
    body = {'data': {'username': 'bob', 'email': 'not-an-email'}}
    httpserver.expect_request('/v2/users/me').respond_with_json(body)

    user = sync_client.user().get()

    assert isinstance(user, UserPublicInfo)
    assert user.username == 'bob'


def test_user_get_raises_response_validation_error_when_no_model_fits(
    httpserver: HTTPServer, sync_client: ApifyClient
) -> None:
    """A user body that fits neither the private nor the public model raises `ResponseValidationError`."""
    httpserver.expect_request('/v2/users/me').respond_with_json(_MISMATCHED_BODY)

    with pytest.raises(ResponseValidationError) as exc_info:
        sync_client.user().get()

    _assert_response_validation_error(exc_info.value, path='/v2/users/me', method='GET')


def test_invalid_request_input_still_raises_pydantic_validation_error(sync_client: ApifyClient) -> None:
    """A validation error on request input is raised as is, since no response is involved."""
    with pytest.raises(ValidationError) as exc_info:
        sync_client.actor('actor-1').update(versions=[{'unexpected': 'shape'}])

    assert not isinstance(exc_info.value, ResponseValidationError)


async def test_user_get_falls_back_to_public_info_async(httpserver: HTTPServer, async_client: ApifyClientAsync) -> None:
    """A user body that fits only the public model still returns the public info."""
    body = {'data': {'username': 'bob', 'email': 'not-an-email'}}
    httpserver.expect_request('/v2/users/me').respond_with_json(body)

    user = await async_client.user().get()

    assert isinstance(user, UserPublicInfo)
    assert user.username == 'bob'


async def test_user_get_raises_response_validation_error_when_no_model_fits_async(
    httpserver: HTTPServer, async_client: ApifyClientAsync
) -> None:
    """A user body that fits neither the private nor the public model raises `ResponseValidationError`."""
    httpserver.expect_request('/v2/users/me').respond_with_json(_MISMATCHED_BODY)

    with pytest.raises(ResponseValidationError) as exc_info:
        await async_client.user().get()

    _assert_response_validation_error(exc_info.value, path='/v2/users/me', method='GET')


class _Contact(BaseModel):
    email: EmailStr


class _Checked(BaseModel):
    value: str

    @field_validator('value')
    @classmethod
    def _check(cls, value: str) -> str:
        raise PydanticCustomError('custom_failure', 'The value {value} is not accepted', {'value': value})


@pytest.mark.parametrize(
    ('model', 'data', 'error_type'),
    [
        pytest.param(_Contact, {'email': 'not-an-email'}, 'value_error', id='email_field'),
        pytest.param(_Checked, {'value': 'x'}, 'custom_failure', id='custom_error'),
    ],
)
def test_validate_response_rebuilds_errors_pydantic_cannot_recreate(
    model: type[BaseModel], data: dict[str, Any], error_type: str
) -> None:
    """Entries that `from_exception_data()` rejects are rebuilt with the same type, message, location and input."""
    with pytest.raises(ResponseValidationError) as exc_info:
        validate_response(model, data)

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValidationError)

    (error,) = exc_info.value.errors(include_url=False)
    (original,) = cause.errors(include_url=False)
    assert error['type'] == error_type
    assert error['msg'] == original['msg']
    assert error['loc'] == original['loc']
    assert error['input'] == original['input']


def test_client_method_clears_request_details_of_earlier_request() -> None:
    """A client method starts without the HTTP method and URL of the request that ran before it."""

    def method(_resource_client: Any) -> tuple[str | None, str | None]:
        return log_context.method.get(), log_context.url.get()

    wrapped = _injects_client_details_to_log_context(method)
    resource_client = Mock(_resource_id='resource-1')

    log_context.method.set('GET')
    log_context.url.set('https://api.apify.com/v2/datasets/ds-0')

    assert wrapped(resource_client) == (None, None)
