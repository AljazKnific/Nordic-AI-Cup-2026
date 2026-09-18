# The prediction endpoint. Ollama runs as its own service alongside this one:
# see docker-compose.yml. Nothing here calls a cloud API, which is a hard rule
# of the competition -- both models run locally, wherever "locally" is.
#
# Build for the host you will deploy to. Azure is x86, this laptop is ARM:
#   docker buildx build --platform linux/amd64 -t medical-appointment:latest .
FROM python:3.11-slim

WORKDIR /app

# ffmpeg-free: faster-whisper reads MP3 through PyAV, which ships its own codecs.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ASR weights into the image so a running container never needs the
# network for transcription, and so the first request does not pay for a
# download. asr.py looks for them in /app/models.
ENV ASR_MODEL=distil-large-v3
RUN python -c "from faster_whisper import WhisperModel; \
    WhisperModel('distil-large-v3', device='cpu', compute_type='int8', \
                 download_root='/app/models')"

# The server and the model. Dev tooling is excluded by .dockerignore.
COPY dtos.py utils.py api.py example.py pipeline.py answering.py asr.py \
     ollama_client.py transcripts.py ./

# Reach Ollama by service name rather than localhost: in Compose they are
# separate containers. Overridable for a single-host or sidecar deployment.
# The weights are baked in above, so never reach for the hub at startup: a
# slow or absent network would otherwise delay or block the server coming up.
ENV OLLAMA_HOST=http://ollama:11434 \
    OLLAMA_MODEL=qwen3:14b \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 9054

# The evaluator gives up on silence, so surface unreadiness as a failed
# healthcheck rather than a hanging request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9054/api', timeout=5)"

CMD ["python", "api.py"]
