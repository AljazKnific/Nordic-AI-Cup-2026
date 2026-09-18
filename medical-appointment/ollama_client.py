"""The local answering model, behind a one-function interface.

Deliberately thin: everything interesting — prompt construction, parsing, spans —
lives in :mod:`answering`, where it can be tested without a model running. This
module only moves bytes.

Calls must stay **sequential**. Concurrent requests evict each other's cached
prompt prefix, which is the whole saving (ADR-0001).
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

# 127.0.0.1, not localhost: Ollama binds IPv4 only, and localhost resolves to
# ::1 first on this machine, which fails with a bare connection refused.
HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
MODEL = os.environ.get('OLLAMA_MODEL', 'qwen3:14b')
TIMEOUT_SECONDS = float(os.environ.get('OLLAMA_TIMEOUT', '30'))

# The answer is a few tokens of JSON plus a short quote. Generation runs at
# ~26 tok/s, so an unbounded reply is a budget risk, not just a slow one.
MAX_TOKENS = 160

# Ollama defaults to a 4096-token context. A transcript prompt runs to ~10,600,
# so the default silently truncates the conversation -- and a question answered
# from a truncated transcript looks like a model failure, not a config one.
CONTEXT_TOKENS = int(os.environ.get('OLLAMA_NUM_CTX', '16384'))

# Ollama unloads an idle model after about five minutes. Waiting in a
# validation queue is exactly that kind of idle, and the reload would be paid
# by the first scored request -- the one with the least budget to spare. Pin
# the model in memory for the duration instead.
KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '60m')


def answer(prompt: str) -> str:
    """Send one prompt, return the model's raw reply text."""
    started = time.perf_counter()
    response = requests.post(
        f'{HOST}/api/generate',
        json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False,
            'think': False,
            'keep_alive': KEEP_ALIVE,
            'options': {
                'temperature': 0.0,
                'num_predict': MAX_TOKENS,
                'num_ctx': CONTEXT_TOKENS,
            },
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    elapsed = time.perf_counter() - started
    generated = payload.get('eval_count') or 0
    logger.info(
        'LLM: %d prompt tokens, %d generated, %.2f s (%.0f tok/s generation)',
        payload.get('prompt_eval_count') or 0,
        generated,
        elapsed,
        generated / elapsed if elapsed else 0.0,
    )
    return payload.get('response', '')


def warm_up() -> None:
    """Load the model into memory before the attempt starts."""
    try:
        started = time.perf_counter()
        requests.post(
            f'{HOST}/api/generate',
            json={'model': MODEL, 'prompt': 'ok', 'stream': False,
                  'think': False, 'keep_alive': KEEP_ALIVE,
                  'options': {'num_predict': 1, 'num_ctx': CONTEXT_TOKENS}},
            timeout=TIMEOUT_SECONDS,
        )
        logger.info('LLM warm-up in %.1f s', time.perf_counter() - started)
    except Exception:
        # A cold model is slow, not fatal. Never let warm-up stop the server
        # from coming up: an endpoint that does not start scores zero.
        logger.exception('LLM warm-up failed; continuing cold')
