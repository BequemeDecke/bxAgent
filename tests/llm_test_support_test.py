"""Unit tests for :mod:`tests.llm_test_support`.

These tests verify the resilience helper *without* hitting a real LLM, so the
skip-vs-fail / retry / bounded-timeout behaviour is checked deterministically and fast.
"""

import asyncio
from unittest import TestCase
from unittest.mock import Mock

import openai

from tests.llm_test_support import LLMTransientError, invoke_with_resilience


def _fake_httpx_response() -> Mock:
    """A minimal stand-in for ``httpx.Response``.

    OpenAI's status errors (``RateLimitError``, ``InternalServerError``, ...) call
    ``response.request`` during construction, so the fake needs a ``request`` attr.
    """
    response = Mock(name="httpx.Response")
    response.request = Mock(name="httpx.Request")
    return response


def _fake_httpx_request() -> Mock:
    """A minimal stand-in for ``httpx.Request`` (used by connection/timeout errors)."""
    return Mock(name="httpx.Request")


def _coro_factory_returning(value):
    """A factory that returns a fresh coroutine yielding ``value`` on each call."""

    async def _coro():
        await asyncio.sleep(0)
        return value

    return _coro


def _coro_factory_raising(exc):
    """A factory that returns a fresh coroutine raising ``exc`` on each call."""

    async def _coro():
        await asyncio.sleep(0)
        raise exc

    return _coro


def _make_coro_factory_with_counter(exc, fail_n_times, then_value):
    """Factory that raises ``exc`` for the first ``fail_n_times`` calls then returns
    ``then_value``. Tracks how many times it was invoked."""

    state = {"calls": 0}

    async def _coro():
        await asyncio.sleep(0)
        state["calls"] += 1
        if state["calls"] <= fail_n_times:
            raise exc
        return then_value

    return _coro, state


class TestInvokeWithResilience(TestCase):
    def test__returns_result_on_first_success(self):
        factory = _coro_factory_returning("ok")

        result = asyncio.run(
            invoke_with_resilience(factory, overall_timeout=5, max_attempts=3)
        )

        self.assertEqual(result, "ok")

    def test__retries_on_transient_error_then_succeeds(self):
        factory, state = _make_coro_factory_with_counter(
            openai.APITimeoutError("timed out"), fail_n_times=1, then_value="recovered"
        )

        result = asyncio.run(
            invoke_with_resilience(
                factory, overall_timeout=5, max_attempts=3, retry_backoff=0
            )
        )

        self.assertEqual(result, "recovered")
        self.assertEqual(state["calls"], 2)  # one failure + one success

    def test__raises_transient_error_after_exhausting_retries(self):
        factory = _coro_factory_raising(openai.APITimeoutError("timed out"))

        with self.assertRaises(LLMTransientError):
            asyncio.run(
                invoke_with_resilience(
                    factory, overall_timeout=5, max_attempts=2, retry_backoff=0
                )
            )

    def test__does_not_retry_non_transient_error(self):
        # An authentication error is NOT transient -> must surface immediately as a real
        # failure (re-raised), not be swallowed into an LLMTransientError/skip.
        auth_error = openai.AuthenticationError(
            "invalid api key", response=_fake_httpx_response(), body=None
        )
        factory, state = _make_coro_factory_with_counter(
            auth_error, fail_n_times=99, then_value="should-not-reach"
        )

        with self.assertRaises(openai.AuthenticationError):
            asyncio.run(
                invoke_with_resilience(
                    factory, overall_timeout=5, max_attempts=3, retry_backoff=0
                )
            )

        # A non-transient error must abort immediately without any retry.
        self.assertEqual(state["calls"], 1)

    def test__treats_rate_limit_and_5xx_and_connection_error_as_transient(self):
        for exc in (
            openai.RateLimitError(
                "slow down", response=_fake_httpx_response(), body=None
            ),
            openai.InternalServerError(
                "boom", response=_fake_httpx_response(), body=None
            ),
            openai.APIConnectionError(request=_fake_httpx_request()),
            openai.APITimeoutError(request=_fake_httpx_request()),
        ):
            with self.subTest(error=type(exc).__name__):
                factory = _coro_factory_raising(exc)
                with self.assertRaises(LLMTransientError):
                    asyncio.run(
                        invoke_with_resilience(
                            factory,
                            overall_timeout=5,
                            max_attempts=2,
                            retry_backoff=0,
                        )
                    )

    def test__overall_timeout_raises_transient_error(self):
        # A coroutine that never completes within the overall budget must cause an
        # LLMTransientError (not hang forever / not a raw asyncio.TimeoutError).
        async def _coro():
            await asyncio.sleep(10)
            return "never"

        with self.assertRaises(LLMTransientError):
            asyncio.run(
                invoke_with_resilience(
                    lambda: _coro(), overall_timeout=0.1, max_attempts=1
                )
            )

    def test__transient_error_is_subclass_of_runtime_error(self):
        # Callers may catch it generically; guarantee it is a RuntimeError.
        self.assertTrue(issubclass(LLMTransientError, RuntimeError))
