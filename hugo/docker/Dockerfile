# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.21@sha256:a8560b36e8b8210634f77d9f7f9efd7ffa463e380b75e2e74aff4511df3ef88c as build
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

FROM docker.io/nginxinc/nginx-unprivileged:1.28@sha256:f78874a7cd6c4dd8a5d33a287c7a09a2a4ae8717fc843500246d5b69b9dcb239

COPY --chown=nginx:nginx --from=build /src/hugo/public /site
COPY --chown=nginx:nginx ./hugo/docker/default.conf /etc/nginx/conf.d/

USER nginx
