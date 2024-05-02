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

FROM ghcr.io/nginxinc/nginx-unprivileged:1.26.0-bookworm@sha256:b1dab22cb3186e156b812aea2e37d216276e62effe6b7c5e7e4b93b6fe72bd22
COPY --from=build /src/public /usr/share/nginx/html
EXPOSE 8080
