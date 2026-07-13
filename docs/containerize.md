# Containerize

The container image is built with [Podman](https://podman.io/).
On macOS, nixpkgs' `podman` bundles vfkit (the Apple `applehv` VM) and gvproxy, so `podman machine` runs on Apple Silicon with no Homebrew, Docker Desktop, or Colima.
The devshell's `podman` is a PATH shim that lazily starts (or offers to `init`) the VM on first use — see [nix.md](./nix.md) and `flake.nix`.

One-time setup (the shim offers to do this for you the first time you run `podman`):

```sh
podman machine init        # creates the applehv VM (downloads a machine image)
podman machine start       # boots it
```

## Local

The `just` recipes wrap the scripts below; prefer them:

```sh
# Build a multi-arch (linux/amd64,linux/arm64) manifest tagged :debug — matches CI.
just hugo build
# ...or build just the native arch for a faster local iteration:
just hugo build platforms="linux/arm64"

# Run the :debug image locally (serves http://127.0.0.1 with an HTTP-friendly nginx conf)
just hugo run
```

Equivalent raw podman commands (run from the repo root):

```sh
# Multi-arch build into a manifest. podman requires --manifest (not --tag) for >1 platform.
# The PAT is passed as a build secret (never a build-arg / image layer).
podman build \
  --platform linux/amd64,linux/arm64 \
  --secret id=treadmill_pat,env=TREADMILL_PAT \
  --manifest ghcr.io/ahgraber/aimlbling-about:debug \
  -f ./hugo/docker/Dockerfile \
  .

# Run the host-arch image from the manifest.
podman run --rm -p 80:80 ghcr.io/ahgraber/aimlbling-about:debug
echo "Navigate to http://127.0.0.1"
```

Cross-arch builds use the podman machine's `qemu-user-static` emulation.
For this Dockerfile that is cheap: the build stage is pinned to `$BUILDPLATFORM` (runs natively), and the runtime stage has no `RUN` instructions, so the non-native variant needs no emulated execution.
If an emulated `RUN` ever does fail with `exec format error`, confirm the emulator is registered inside the VM:

```sh
podman machine ssh 'ls /proc/sys/fs/binfmt_misc/ | grep qemu'
# if missing on an older machine image:
podman machine ssh 'sudo rpm-ostree install qemu-user-static && sudo systemctl reboot'
```

## Push debug image to github

```sh
# Login to the GitHub container registry (paste a Personal Access Token when prompted).
podman login ghcr.io --username ahgraber

# Build the multi-arch manifest, then push all arches.
podman build \
  --platform linux/amd64,linux/arm64 \
  --secret id=treadmill_pat,env=TREADMILL_PAT \
  --manifest ghcr.io/ahgraber/aimlbling-about:debug \
  -f ./hugo/docker/Dockerfile \
  .
podman manifest push --all \
  ghcr.io/ahgraber/aimlbling-about:debug \
  docker://ghcr.io/ahgraber/aimlbling-about:debug

# Pull and run it locally to verify.
podman pull ghcr.io/ahgraber/aimlbling-about:debug
podman run --rm -p 80:80 ghcr.io/ahgraber/aimlbling-about:debug
```

> The published `latest` image is built and pushed by CI
> (`.github/workflows/publish.yaml`) on merges to `main`; the manual flow above is for
> debugging.
