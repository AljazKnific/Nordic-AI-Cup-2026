"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI, Request

# Before importing `example`, which loads and exercises both models on the way
# in. Configure logging after it and every warm-up timing is discarded -- which
# is exactly the number you want when a slow start is the thing under suspicion.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
)
logger = logging.getLogger(__name__)

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto  # noqa: E402
from example import predict  # noqa: E402
from utils import validate_response  # noqa: E402

HOST = '0.0.0.0'
PORT = 9054

app = FastAPI()
start_time = time.time()


@app.middleware('http')
async def stamp_arrival(request: Request, call_next):
    """Record when the request landed, before its body has been read.

    The evaluator's 60 s starts when it sends, ours used to start once FastAPI
    had already received and parsed a ~5 MB base64 body. On 2026-09-19 that gap
    was 25 s on one conversation: answered in 35 s by our clock, scored as a
    timeout by theirs. Everything downstream budgets from this reading.
    """
    request.state.arrived = time.perf_counter()
    response = await call_next(request)
    if request.url.path == '/predict':
        logger.info('REQUEST served in %.1f s wall, arrival to response',
                    time.perf_counter() - request.state.arrived)
    return response


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto, raw: Request):
    """Answer every question about one conversation."""
    response = predict(request, arrived=getattr(raw.state, 'arrived', None))

    # Fail here, loudly, rather than having the evaluator silently score every
    # question about this conversation wrong.
    validate_response(response, expected_count=len(request.questions))

    return response


@app.get('/api')
def hello():
    return {
        'service': 'medical-appointment-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    # **Never close an idle connection before the evaluator has given up on
    # one.** uvicorn's default keep-alive is 5 s and the evaluator's requests
    # arrive 25-45 s apart, so the server closed every connection between them.
    # A client holding that connection in a pool and writing its next POST as
    # the close lands loses the request outright: it never reaches the ASGI app
    # and nothing here logs an error, because a keep-alive close is routine.
    #
    # That is the signature seen on 2026-09-20 -- sample_45 absent from the log
    # entirely, one 61.6 s gap between two ordinary requests, no exception --
    # and on 2026-09-19, when sample_46 vanished the same way. A lost
    # conversation is all ten of its marks.
    #
    # 75 s, above the evaluator's own 60 s timeout, so the close is always the
    # client's to make.
    uvicorn.run('api:app', host=HOST, port=PORT, timeout_keep_alive=75)
