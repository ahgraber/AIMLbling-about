# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.19@sha256:c5b1261d6d3e43071626931fc004f70149baeba2c8ec672bd4f27761f8e1ad6b as build
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

FROM ghcr.io/nginxinc/nginx-unprivileged:1.25.5-bookworm@sha256:8d3f54fc88575ada35543c7db06a4c2409aa46425f4a97f3429e6726a883e837
COPY --from=build /src/public /usr/share/nginx/html
EXPOSE 8080
