FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# copy project + install it WITHOUT changing dependency set
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN pip install --no-deps .

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "flightright.api.app:app", "--host", "0.0.0.0", "--port", "8080"]