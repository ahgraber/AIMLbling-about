# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.19 as build
RUN \
    apk add --no-cache \
      git \
      hugo \
      tzdata \
 && apk add --no-cache dart-sass --repository=http://dl-cdn.alpinelinux.org/alpine/edge/testing/
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/tmp/hugo_cache \
    hugo

FROM ghcr.io/nginxinc/nginx-unprivileged:1.25.5-bookworm
COPY --from=build /src/public /usr/share/nginx/html
EXPOSE 8080
