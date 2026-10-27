# The UI: Agent Chat UI, unmodified, built from a pinned commit.
FROM node:22.23.4-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ui
# Agent Chat UI publishes no releases, tags or images, so a commit is the only fixed
# version: this is main as of 2026-10-19. Update it deliberately, then rebuild and test.
ARG AGENT_CHAT_UI_COMMIT=7a503412aeeb4534a24b04bd97b618ee3871821e
RUN git init --quiet \
    && git fetch --quiet --depth 1 https://github.com/langchain-ai/agent-chat-ui.git "$AGENT_CHAT_UI_COMMIT" \
    && git checkout --quiet FETCH_HEAD

ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable && pnpm install --frozen-lockfile

# Baked into the browser bundle at build time. The browser talks to the UI's own /api
# passthrough, which forwards to the agent (LANGGRAPH_API_URL, set at run time), so only
# port 3000 needs to be reachable.
ARG NEXT_PUBLIC_API_URL=http://localhost:3000/api
ARG NEXT_PUBLIC_ASSISTANT_ID=triage
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL NEXT_PUBLIC_ASSISTANT_ID=$NEXT_PUBLIC_ASSISTANT_ID
RUN pnpm build

EXPOSE 3000
CMD ["pnpm", "start"]
