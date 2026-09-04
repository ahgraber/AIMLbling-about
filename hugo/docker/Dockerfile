# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

FROM --platform=$BUILDPLATFORM docker.io/library/golang:1.26-alpine3.23@sha256:33ce311e5eecedee48ec1b84419c1306e9fbd71009f0d5c3f2a6904b579c1ecc AS build

COPY --from=uv /uv /uvx /bin/

# Set shell options for safety
SHELL ["/bin/ash", "-eo", "pipefail", "-c"]

ARG BUILDARCH

RUN  apk add --no-cache \
      git \
      hugo \
      tzdata \
      curl \
      unzip

# Set Dart Sass version
ENV DART_SASS_VERSION=1.86.3

# Download the Dart Sass binary for the architecture that executes this build stage.
RUN case "${BUILDARCH}" in \
      amd64) sass_arch="x64" ;; \
      arm64) sass_arch="arm64" ;; \
      *) echo "unsupported build architecture for Dart Sass: ${BUILDARCH}" >&2; exit 1 ;; \
    esac && \
    curl -sSL https://github.com/sass/dart-sass/releases/download/${DART_SASS_VERSION}/dart-sass-${DART_SASS_VERSION}-linux-${sass_arch}-musl.tar.gz \
    | tar -xz -C /usr/local/bin --strip-components=1

ENV UV_PYTHON=3.12.13
ENV UV_MANAGED_PYTHON=1
RUN uv python install --no-bin "${UV_PYTHON}"
ENV UV_PYTHON_DOWNLOADS=never
ENV PYTHONPATH=/src/site-search/src

WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY site-search/pyproject.toml ./site-search/pyproject.toml

# Alpine uses musl and cannot consume glibc-only manylinux wheels. Keep --no-build:
# dependencies must provide a musllinux wheel or move the build stage to glibc instead of
# silently pulling compiler, Rust, or BLAS toolchains during an upgrade.
RUN uv sync --frozen --package site-search --no-default-groups --no-install-project --no-build

COPY .git ./.git
COPY site-search/src ./site-search/src
COPY hugo ./hugo

RUN --mount=type=secret,id=treadmill_pat,required=false \
    --mount=type=cache,target=/tmp/hugo_cache \
    if [ -s /run/secrets/treadmill_pat ]; then \
      treadmill_pat="$(cat /run/secrets/treadmill_pat)"; \
      git config --global url."https://${treadmill_pat}:x-oauth-basic@github.com/".insteadOf "https://github.com/"; \
    fi && \
    rm -rf ./hugo/public ./hugo/assets/search ./hugo/static/search && \
    hugo -s ./hugo && \
    ./.venv/bin/python -m site_search.gate ./hugo/public/en.search-data.json && \
    ./.venv/bin/python -m site_search.build_index \
      --search-data ./hugo/public/en.search-data.json \
      --out ./hugo/assets/search && \
    hugo --minify -s ./hugo

FROM docker.io/nginxinc/nginx-unprivileged:1.31@sha256:f4a6f1ea26f4e500265b60b1935f2a5a0bc0cc5da4ca02ec32b2099426612229

# Add build-time ARGs
ARG BUILD_DATE
ARG VCS_REF

# Add runtime labels
LABEL org.opencontainers.image.source="https://github.com/ahgraber/aimlbling-about"
LABEL org.opencontainers.image.description="Generate AIMLbling-About static site with Hugo"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"

COPY --chown=nginx:nginx --from=build /src/hugo/public /site
COPY --chown=nginx:nginx --from=build /src/hugo/docker/default.conf /etc/nginx/conf.d/

USER nginx
