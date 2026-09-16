import asyncio
import logging
from typing import Awaitable, Callable

import openai
from langchain.chat_models import init_chat_model

from mdeagent.config import Config

logger = logging.getLogger(__name__)


def build_base_model():
    """Builds the base model using the loaded configuration."""
    agent_config = Config.get_instance().MODEL
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 1
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.BASE_MODEL,
        request_timeout=agent_config.REQUEST_TIMEOUT,
        max_retries=agent_config.MAX_RETRIES,
    )


def build_coding_model():
    """Builds the coding model using the loaded configuration."""
    agent_config = Config.get_instance().MODEL
    return init_chat_model(
        model_provider="openai",  # TODO: Make this configurable later on; Counter: 2
        base_url=agent_config.BASE_URL,
        api_key=agent_config.API_KEY.get_secret_value(),
        model=agent_config.CODING_MODEL,
        request_timeout=agent_config.REQUEST_TIMEOUT,
        max_retries=agent_config.MAX_RETRIES,
    )


"""Helpers for integration tests that drive a *real* LLM.

LLM calls — in particular ``with_structured_output`` on a complex Pydantic schema —
are inherently flaky: the endpoint sometimes takes longer than the configured request
timeout or returns a transient error (timeout, connection reset, rate limit, 5xx) even
when the surrounding code is perfectly correct.

Without protection this flakiness surfaces in the test suite as either

* a long, silent hang (the SDK retries a short request timeout several times, each time
  waiting for the full timeout to elapse), or
* a cryptic ``OpenAITimeoutError`` traceback that fails a test whose logic is fine.

This module provides :func:`invoke_with_resilience`, a thin wrapper that:

1. caps the *total* wall-clock time spent in the LLM call so a hanging endpoint can
   never stall the test indefinitely,
2. retries a bounded number of times on *transient* errors (timeouts, connection
   errors, rate limits, 5xx) with a short backoff and clear logging, and
3. raises :class:`LLMTransientError` once the retries are exhausted so the caller can
   turn an "endpoint unavailable" situation into a *skipped* test instead of a failure.

Non-transient errors (e.g. authentication failures, schema validation errors) are
re-raised immediately and therefore surface as real test failures.
"""


class LLMTransientError(RuntimeError):
    """Raised when an LLM call fails repeatedly due to *transient* infrastructure
    issues (timeouts, connection errors, rate limits, 5xx).

    A transient failure is not a logic failure: callers typically convert it into a
    skipped test rather than a failing one.
    """


# OpenAI errors that indicate a flaky/slow endpoint rather than a broken call.
# Order matters only for readability; ``isinstance`` checks them in order.
#
# * ``APITimeoutError``        — request took longer than the configured timeout.
# * ``APIConnectionError``     — network/DNS/TCP issue reaching the endpoint.
# * ``RateLimitError``         — 429; the gateway asked us to slow down.
# * ``InternalServerError``    — 5xx; the endpoint is having a bad time.
#
# ``OpenAITimeoutError`` (langchain-openai) subclasses ``APITimeoutError``, so it is
# covered automatically.
_TRANSIENT_OPENAI_ERRORS: tuple[type[BaseException], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


async def invoke_with_resilience[T](
    coro_factory: Callable[[], Awaitable[T]],
    *,
    overall_timeout: float = 180.0,
    max_attempts: int = 2,
    retry_backoff: float = 2.0,
) -> T:
    """Await an async LLM call with a bounded overall timeout and transient retry.

    Args:
        coro_factory: A zero-argument callable returning a *fresh* coroutine to await.
            It must be a factory (not an already-created coroutine) so every retry can
            re-create the coroutine — awaiting the same coroutine twice is not possible.
        overall_timeout: Hard cap on the *total* wall-clock seconds across all attempts.
            If this fires, ``LLMTransientError`` is raised (the endpoint is unresponsive).
        max_attempts: Number of attempts before giving up on transient errors.
        retry_backoff: Seconds to sleep before the first retry; doubled after each retry.

    Returns:
        The result of the awaited coroutine.

    Raises:
        LLMTransientError: If every attempt failed with a transient error, or the overall
            timeout fired. Convert this into a skipped test in integration tests.
        Any non-transient exception raised by ``coro_factory`` is re-raised immediately.
    """
    # Prefer a monotonic clock so wall-clock changes cannot skew the budget.
    deadline = asyncio.get_event_loop().time() + overall_timeout

    async def _run_all_attempts() -> T:
        last_exc: BaseException | None = None
        backoff = retry_backoff
        for attempt in range(1, max_attempts + 1):
            # Bail out early if there is no time left for another full attempt — this
            # avoids starting a slow call that would be cancelled immediately by the
            # outer ``asyncio.wait_for``.
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0 and attempt > 1:
                break

            try:
                return await coro_factory()
            except _TRANSIENT_OPENAI_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d failed with transient error (%s): %s",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                )
            except TimeoutError as exc:
                # A bare asyncio.TimeoutError raised from inside the coroutine (e.g. a
                # nested ``wait_for``) is also transient.
                last_exc = exc
                logger.warning(
                    "LLM call attempt %d/%d timed out internally.",
                    attempt,
                    max_attempts,
                )

            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff *= 2

        raise LLMTransientError(
            f"LLM call failed after {max_attempts} attempt(s) due to transient errors; "
            f"the endpoint appears to be slow or temporarily unavailable."
        ) from last_exc

    try:
        return await asyncio.wait_for(_run_all_attempts(), timeout=overall_timeout)
    except TimeoutError as exc:
        raise LLMTransientError(
            f"LLM call did not complete within the overall timeout of "
            f"{overall_timeout:.0f}s — the endpoint appears to be unresponsive."
        ) from exc
