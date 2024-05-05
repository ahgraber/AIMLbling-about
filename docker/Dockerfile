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
    hugo --minify

# FROM ghcr.io/nginxinc/nginx-unprivileged:1.26.0-bookworm@sha256:b1dab22cb3186e156b812aea2e37d216276e62effe6b7c5e7e4b93b6fe72bd22
FROM docker.io/library/nginx:1.26
COPY --chown=nginx:nginx --from=build /src/public /site
COPY --chown=nginx:nginx ./docker/default.conf /etc/nginx/conf.d/
# RUN \
#     bash -c 'mkdir -p /var/cache/nginx/{client_temp,proxy_temp_path,fastcgi_temp,uwsgi_temp,scgi_temp}' \
#     chown -R nginx:nginx /var/cache/nginx/

USER nginx
EXPOSE 80
