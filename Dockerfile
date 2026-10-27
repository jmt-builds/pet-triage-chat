# The agent: the triage graph, served by `langgraph dev` on port 2024.
FROM python:3.12.15-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY langgraph.json ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY evals ./evals
ENV PATH="/app/.venv/bin:$PATH"

# Run as an ordinary user, not root. It owns /app because langgraph keeps its state there.
RUN useradd --no-create-home app && chown app /app
USER app

EXPOSE 2024
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser", "--no-reload"]
