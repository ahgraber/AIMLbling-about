# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/golang:1.26-alpine3.23@sha256:2389ebfa5b7f43eeafbd6be0c3700cc46690ef842ad962f6c5bd6be49ed82039 AS build

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
      hugo \
      tzdata \
      curl \
      unzip

# Set Dart Sass version
ENV DART_SASS_VERSION=1.86.3

# Download and install Dart Sass binary
RUN curl -sSL https://github.com/sass/dart-sass/releases/download/${DART_SASS_VERSION}/dart-sass-${DART_SASS_VERSION}-linux-x64-musl.tar.gz \
    | tar -xz -C /usr/local/bin --strip-components=1

WORKDIR /src
COPY .git ./.git
COPY hugo ./hugo

RUN --mount=type=secret,id=treadmill_pat,required=false \
    --mount=type=cache,target=/tmp/hugo_cache \
    if [ -s /run/secrets/treadmill_pat ]; then \
      treadmill_pat="$(cat /run/secrets/treadmill_pat)"; \
      git config --global url."https://${treadmill_pat}:x-oauth-basic@github.com/".insteadOf "https://github.com/"; \
    fi && \
    hugo --minify -s ./hugo

FROM docker.io/nginxinc/nginx-unprivileged:1.29@sha256:a4b4d6c0ea8ecf5af39ca16ffd0b388aa3afd66108883560f78adb13e84d193e

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
