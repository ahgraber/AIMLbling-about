{
  description = "A simple flake to install Hugo test environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devshell.url = "github:numtide/devshell";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, devshell, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system: {
      devShells.default =
        let
          pkgs = import nixpkgs {
            inherit system;
            # bring devshell attribute into the pkgs
            overlays = [ devshell.overlays.default ];
          };

          # --- container image build/run (`just hugo build` / `just hugo run`) ---
          #
          # Engine: Podman. nixpkgs' podman bundles vfkit (Apple applehv VM) + gvproxy
          # (VM networking), so `podman machine init/start` works on Apple Silicon with no
          # Homebrew, Docker Desktop, or Colima. The podman machine-os image ships
          # qemu-user-static, so emulated cross-arch builds
          # (`podman build --platform linux/amd64,linux/arm64 --manifest ...`) work from an
          # arm64 host — matching the linux/amd64+linux/arm64 image CI publishes. (This
          # site's Dockerfile pins its build stage to $BUILDPLATFORM and its runtime stage
          # has no RUN steps, so the amd64 variant needs no emulated execution at all.)
          #
          #   One-time setup (the shim below offers to run the init for you):
          #     podman machine init        # creates the applehv VM
          #     podman machine start       # boots it (exposes a Docker-compat socket)
          #
          # Lazy bring-up: on darwin `podman` is a PATH shim so the FIRST command that needs
          # the VM ensures its API socket is live — (re)starting the machine when the socket
          # is missing, or offering to `init` one when no machine exists. Nothing touches the
          # VM merely by entering the shell. macOS reaps the socket from /tmp after ~3 days
          # (leaving the VM "running" but unreachable), so the shim checks the socket itself,
          # not just `podman machine` state. A PATH shim (not a shellHook function) so the
          # lazy bring-up works in any login shell — direnv exports env vars to zsh, but not
          # bash functions. The shim references the real podman by absolute store path so it
          # is the only `podman` on PATH (no collision, and it deterministically wins over a
          # system install).
          podmanReal = "${pkgs.podman}/bin/podman";
          podmanShim = pkgs.writeShellScriptBin "podman" ''
            # Pass machine management straight through (no ensure, no recursion).
            if [ "''${1:-}" = "machine" ]; then exec ${podmanReal} "$@"; fi
            sock="$(${podmanReal} machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}' 2>/dev/null || true)"
            if [ -z "$sock" ] || [ ! -S "$sock" ]; then
              state="$(${podmanReal} machine inspect --format '{{.State}}' 2>/dev/null || true)"
              if [ -z "$state" ]; then
                # No machine at all: offer to create one (init downloads a VM image, one-time).
                # Only prompt when interactive; non-interactive callers (CI, subprocesses) get
                # the hint and bail rather than seeing an unanswerable question.
                if ! { : < /dev/tty; } 2>/dev/null; then
                  echo "aimlbling: no podman machine — run 'podman machine init && podman machine start' once" >&2
                  exit 1
                fi
                printf 'aimlbling: no podman machine. Initialize one now? It downloads a VM image (one-time). [Y/n] ' > /dev/tty
                read -r reply < /dev/tty || reply=n
                # Fail safe: only an affirmative reply (or bare Enter) triggers the heavy init.
                case "$reply" in
                  "" | [Yy]*) ;;
                  *) echo "aimlbling: skipped — run 'podman machine init && podman machine start' when ready" >&2; exit 1 ;;
                esac
                echo "aimlbling: initializing podman machine (this can take a few minutes)..." >&2
                ${podmanReal} machine init || { echo "aimlbling: 'podman machine init' failed" >&2; exit 1; }
                ${podmanReal} machine start >/dev/null 2>&1 || { echo "aimlbling: 'podman machine start' failed" >&2; exit 1; }
              else
                # Machine exists but its socket is gone (macOS reaps it): (re)start it.
                echo "aimlbling: podman machine socket unavailable — (re)starting..." >&2
                [ "$state" = "running" ] && ${podmanReal} machine stop >/dev/null 2>&1
                ${podmanReal} machine start >/dev/null 2>&1 || { echo "aimlbling: 'podman machine start' failed" >&2; exit 1; }
              fi
            fi
            exec ${podmanReal} "$@"
          '';
        in
        pkgs.devshell.mkShell {
          name = "hugo-devshell";
          # a list of packages to add to the shell environment
          packages = [
            pkgs.hugo
            pkgs.dart-sass
          ]
          # Container image build/run uses Podman (see the shim above). On darwin the lazy
          # shim is the only `podman` on PATH so it always brings up the applehv VM before
          # the real binary runs; on Linux podman runs natively with no machine.
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ podmanShim ]
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.podman ];
        };
    });
}
