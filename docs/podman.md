# podman

## Containerize

Start podman:

```sh
podman machine start
```

Build the docker image:

```sh
# build image
podman build \
  -t ghcr.io/ahgraber/aimlbling-about:$(date +"%Y.%m.%d-%H%M") \
  -t ghcr.io/ahgraber/aimlbling-about:latest \
  .
```

Run image

```sh
podman machine start
podman run --rm -p 8080:8080 ghcr.io/ahgraber/aimlbling-about:latest
```

## Debug

### `Error: dial unix /tmp/podman/qmp_podman-machine-default.sock: connect: no such file or directory`

```sh
podman machine stop
podman machine rm -f
podman machine init
podman machine start
```

```sh
podman system connection default podman-machine-default
unset SSH_AUTH_SOCK
```
