# syntax=docker/dockerfile:1

# ---- Builder: install dependencies into a venv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Install the CPU build of torch first to keep the image small, then the rest.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Install the package itself.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# ---- Runtime: minimal image with just the venv and source ----
FROM python:3.12-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER appuser

# FRED_API_KEY must be supplied at runtime, e.g.:
#   docker run --rm -e FRED_API_KEY=xxx fred-forecasting --series-id DEXUSEU --epochs 50
ENTRYPOINT ["fred-forecast"]
CMD ["--help"]
