
ARG PYTHON_VERSION=3.10.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:/app/backend" 

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

USER appuser


COPY DataHandle app/DataHandle
COPY backend app/backend

COPY . .

EXPOSE 8080

ENTRYPOINT [ "python", "backend/email-api-app.py" ]
