# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv

FROM --platform=$BUILDPLATFORM docker.io/library/golang:1.26-alpine3.23@sha256:622e56dbc11a8cfe87cafa2331e9a201877271cbff918af53d3be315f3da88cc AS build

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
    rm -rf ./hugo/public ./hugo/static/search && \
    hugo -s ./hugo && \
    ./.venv/bin/python -m site_search.gate ./hugo/public/en.search-data.json && \
    ./.venv/bin/python -m site_search.build_index \
      --search-data ./hugo/public/en.search-data.json \
      --out ./hugo/static/search && \
    hugo --minify -s ./hugo

FROM docker.io/nginxinc/nginx-unprivileged:1.31@sha256:0a7658b1a51df663ad17cfe82ab6edd7f56f82cfe7b49f282f83a30617f3f404

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
