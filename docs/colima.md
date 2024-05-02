# [abiosoft/colima: Container runtimes on macOS (and Linux) with minimal setup](https://github.com/abiosoft/colima)

## Containerize

Start colima:

```sh
colima start
```

Build the container image:

```sh
# build image
docker build \
  -t ghcr.io/ahgraber/aimlbling-about:$(date +"%Y.%m.%d-%H%M") \
  -t ghcr.io/ahgraber/aimlbling-about:latest \
  .
```

Run image:

```sh
docker run --rm -p 8080:8080 ghcr.io/ahgraber/aimlbling-about:latest
```

Stop colima:

```sh
colima stop
```
