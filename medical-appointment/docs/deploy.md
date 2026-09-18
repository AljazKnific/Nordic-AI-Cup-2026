# Deploying the endpoint

The evaluator POSTs one conversation to `http://<host>:9054/predict` and gives it
60 seconds, **averaged across the whole attempt**. Both models must run on the
host: no cloud API may sit in the request path. That is a competition rule, and
it is the reason Ollama is deployed beside the API rather than called remotely.

## Shape

Two containers, defined in `docker-compose.yml`:

- **ollama** — the answering model, in a named volume so it survives rebuilds.
  Not published to the host; only the API container reaches it.
- **api** — this repo. The ASR weights are baked into the image at build time,
  so transcription never needs the network.

A one-shot `model-puller` service fetches `qwen3:14b` on first start (~9 GB) and
exits. Later starts reuse the volume.

```
docker compose up --build      # first run pulls the answering model
curl localhost:9054/api        # readiness
```

## Building for Azure

This laptop is ARM and Azure is x86, so the image must be built for the target:

```
docker buildx build --platform linux/amd64 -t medical-appointment:latest .
```

## Sizing, which is the part that actually decides the score

Every timing we have was measured on an M4 Pro: **39.4 s per conversation
through HTTP** against a 60 s budget, of which ~20 s is ASR and ~13 s is ten
sequential calls to a 14B model. That leaves roughly 20 s of headroom, and a
slower host spends it immediately.

A general-purpose Azure VM without a GPU will not match an M4 Pro on a 14B
model. Before committing to one, measure on the target rather than assuming,
and treat these as the levers, in the order worth trying:

1. **A GPU VM (NC/NV series).** Keeps the current models and removes the
   problem. It also unlocks `ASR_DEVICE=cuda` with `ASR_COMPUTE_TYPE=float16`,
   which is several times faster than int8 on CPU.
2. **A smaller answering model.** `OLLAMA_MODEL=qwen3:8b` needs no code change.
   Accuracy is currently 0.990 with plenty of margin, and answering is the half
   that shrinks; re-score with `tools/score.py` before trusting it.
3. **`ANSWER_DEADLINE`.** Lower it so a slow host returns poor answers instead
   of silence. Five consecutive timeouts end an attempt outright, and a guess is
   worth half a mark where silence is worth nothing.

Everything above is an environment variable. No code changes.

## Memory

`qwen3:14b` needs ~9.3 GB resident, plus ~1.5 GB for the ASR model and the
server. Size the host at 16 GB or more. Note that Docker Desktop on a Mac has
its own memory ceiling, separate from the machine's, and the default is often
too small to run the answering model locally.

## Before an attempt

- Confirm the endpoint answers from **outside** the host, path included.
- Confirm `docker compose ps` shows both services healthy.
- Warm both models before the first request: the API container does this at
  import, so start it and wait for `/api` to respond before submitting.
- Ollama unloads an idle model after a few minutes; the attempt's first request
  then pays the reload. Keep it warm if the attempt is not imminent.
