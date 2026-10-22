FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.13-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group web --no-install-project --no-editable
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group web --no-editable

FROM python:3.13-slim-bookworm AS runtime
ARG APP_VERSION=development
ARG BUILD_SHA=development
LABEL org.opencontainers.image.title="genui-quality-gate" \
    org.opencontainers.image.source="https://github.com/maryshirk/genui-quality-gate" \
    org.opencontainers.image.version=$APP_VERSION \
    org.opencontainers.image.revision=$BUILD_SHA
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    XDG_CACHE_HOME=/tmp/cache \
    GENUI_BUILD_SHA=$BUILD_SHA
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /state && chown app:app /state
COPY --from=builder /app/.venv /app/.venv
USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).close()"]
CMD ["uvicorn", "genui_quality_gate.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
