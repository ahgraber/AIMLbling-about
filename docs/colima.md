# Containerize

Using [abiosoft/colima: Container runtimes on macOS (and Linux) with minimal setup](https://github.com/abiosoft/colima)

## Local

Start colima:

```sh
colima start
```

Build the container image:

```sh
# build image
docker build -t ghcr.io/ahgraber/aimlbling-about:debug -f ./docker/Dockerfile .
```

Run image:

```sh
docker run --rm -p  80:80 ghcr.io/ahgraber/aimlbling-about:debug
```

Stop colima:

```sh
colima stop
```

## Push debug image to github

Start colima:

```sh
colima start
```

Login to github container registry:

```sh
docker login ghcr.io --username ahgraber
# paste Personal Access Token
```

Build and push image:

```sh
docker buildx create --use
docker buildx build \
  --push \
  --platform  linux/amd64,linux/arm64 \
  -t ghcr.io/ahgraber/aimlbling-about:debug \
  -f ./docker/Dockerfile \
  .
```

Run image locally:

```sh
docker pull ghcr.io/ahgraber/aimlbling-about:debug
docker run --rm -p 80:80 ghcr.io/ahgraber/aimlbling-about:debug
```

Stop colima:

```sh
colima stop
```
