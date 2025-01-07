# https://sko.ai/blog/how-to-actually-build-hugo-containers/
FROM --platform=$BUILDPLATFORM docker.io/library/alpine:3.21@sha256:b97e2a89d0b9e4011bb88c02ddf01c544b8c781acf1f4d559e7c8f12f1047ac3 as build
RUN  apk add --no-cache \
      git \
      hugo \
      tzdata \
  && apk add --no-cache dart-sass --repository=https://dl-cdn.alpinelinux.org/alpine/edge/testing/
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/tmp/hugo_cache \
    hugo --minify -s ./hugo

FROM docker.io/library/nginx:1.27@sha256:42e917aaa1b5bb40dd0f6f7f4f857490ac7747d7ef73b391c774a41a8b994f15
# implement changes required to run NGINX as an less-privileged user
RUN  sed -i 's,/var/run/nginx.pid,/tmp/nginx.pid,' /etc/nginx/nginx.conf \
  && sed -i "/^http {/a \    proxy_temp_path /tmp/proxy_temp;\n    client_body_temp_path /tmp/client_temp;\n    fastcgi_temp_path /tmp/fastcgi_temp;\n    uwsgi_temp_path /tmp/uwsgi_temp;\n    scgi_temp_path /tmp/scgi_temp;\n" /etc/nginx/nginx.conf \
  # nginx user must own the cache and etc directory to write cache and tweak the nginx config
  && chown -R 101:0 /var/cache/nginx \
  && chmod -R g+w /var/cache/nginx \
  && chown -R 101:0 /etc/nginx \
  && chmod -R g+w /etc/nginx

COPY --chown=nginx:nginx --from=build /src/hugo/public /site
COPY --chown=nginx:nginx ./hugo/docker/default.conf /etc/nginx/conf.d/

USER nginx
