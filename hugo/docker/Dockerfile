# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.22@sha256:4b7ce07002c69e8f3d704a9c5d6fd3053be500b7f1c69fc0d80990c2ad8dd412 AS build

# Set shell options for safety
SHELL ["/bin/ash", "-eo", "pipefail", "-c"]

# Add build-time ARGs
ARG BUILD_DATE
ARG VCS_REF

# Add build-time labels
LABEL org.opencontainers.image.source="https://github.com/ahgraber/aimlbling-about"
LABEL org.opencontainers.image.description="Generate AIMLbling-About static site with Hugo"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"

RUN  apk add --no-cache \
      git \
      go \
      hugo \
      tzdata \
      curl \
      unzip

# Set Dart Sass version
ENV DART_SASS_VERSION=1.86.3

# Download and install Dart Sass binary
RUN curl -sSL https://github.com/sass/dart-sass/releases/download/${DART_SASS_VERSION}/dart-sass-${DART_SASS_VERSION}-linux-x64-musl.tar.gz \
    | tar -xz -C /usr/local/bin --strip-components=1

ARG TREADMILL_PAT
RUN git config --global url."https://${TREADMILL_PAT}:x-oauth-basic@github.com/".insteadOf "https://github.com/"

WORKDIR /src
COPY . .

RUN --mount=type=cache,target=/tmp/hugo_cache \
    hugo --minify -s ./hugo

FROM docker.io/nginxinc/nginx-unprivileged:1.29@sha256:d62638f75536e8b6be939bdfc6bb17d99b167f0c2af4955440bc7314dc72f5a0

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
COPY --chown=nginx:nginx ./hugo/docker/default.conf /etc/nginx/conf.d/

USER nginx
