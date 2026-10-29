# Containerize

The site image is built from `hugo/docker/Dockerfile`.
CI (`.github/workflows/publish.yaml`) builds it with `docker buildx` and publishes `ghcr.io/ahgraber/aimlbling-about:latest` on merges to `main`.
The local recipes build and push the `:debug` tag.

## Local recipes

```sh
# Build the native platform into the local image store.
just hugo build
# ...or another platform:
just hugo build linux/amd64

# Run the :debug image (serves http://127.0.0.1:8081 with an HTTP-friendly nginx conf).
just hugo run

# Build linux/amd64 and linux/arm64 and push both to ghcr.io (log in first).
just hugo push
```

`build` loads one platform only.
A multi-platform image loads only into a store with Docker's containerd image store, which socktainer and podman's Docker API do not have.
`push` therefore uploads straight from the builder and never loads the image locally.

The build passes `TREADMILL_PAT` from the environment or `.env` as a build secret, never as a build arg.

## Container CLI

The recipes drive a container CLI, not a runtime.
`scripts/container_engine.sh` selects `docker` if it is on `PATH`, else `podman`.
Set `AIMLBLING_ENGINE=docker` or `AIMLBLING_ENGINE=podman`, in the shell or in `.env`, to choose.

Which runtime the CLI reaches is host configuration, and the Nix devshell provides no container tools.
On macOS, the home-manager module in `~/.nixconfig/modules/home/dev/containers/` routes `docker` and `podman` to Apple's container runtime through socktainer, with a podman machine as the fallback; its README covers setup.
On Linux, install podman or Docker.

Each CLI is one file in `scripts/engines/`, named after the CLI.
To add one, add `scripts/engines/<cli>.sh` with the functions that `scripts/container_engine.sh` documents, and add the CLI to `_engine_candidates`.

### docker

Builds use `docker buildx build`, so they need a BuildKit builder for `--secret`.
The macOS home-manager module creates one; Docker Engine has one by default.

### podman

Builds pass `BUILDARCH` explicitly.
`just hugo push` builds a manifest and runs `podman manifest push --all`.
`build` and `push` share the `:debug` name, so each one first deletes the image or manifest that the other left.
With the macOS home-manager module, `podman manifest` and `podman build --manifest` work only on podman's VM, so run `AIMLBLING_ENGINE=podman CONTAINERS_BACKEND=podman just hugo <recipe>`.
The module's default (`docker` on Apple's runtime) needs no setting.

## Raw commands

Equivalent to `just hugo push` with docker (run from the repo root):

```sh
docker login ghcr.io --username ahgraber
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --secret id=treadmill_pat,env=TREADMILL_PAT \
  --push \
  --tag ghcr.io/ahgraber/aimlbling-about:debug \
  -f ./hugo/docker/Dockerfile \
  .
```

Leave out `--secret` when `TREADMILL_PAT` is unset.

The Dockerfile pins its build stage to `$BUILDPLATFORM`, and its runtime stage has no `RUN` instructions.
So the non-native platform needs no emulated execution.
