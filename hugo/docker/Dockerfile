# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.23@sha256:25109184c71bdad752c8312a8623239686a9a2071e8825f20acb8f2198c3f659 AS build

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

FROM docker.io/nginxinc/nginx-unprivileged:1.29@sha256:5366aa1ef633415d3fb30f8b5b2d824c4ae1c231a0f63358e7f39734fa5894c0

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
