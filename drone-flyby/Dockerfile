FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The detector and motion fallback load reference crops and annotations from
# src/helsinki at startup, so they must be present in the deployed image.
COPY dtos.py utils.py example.py api.py ./
COPY src ./src

EXPOSE 9053

CMD ["python", "api.py"]
