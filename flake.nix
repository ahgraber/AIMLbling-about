{
  description = "A simple flake to install Hugo test environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devshell.url = "github:numtide/devshell";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      devshell,
      flake-utils,
      nixpkgs,
    }:
    flake-utils.lib.eachDefaultSystem (system: {
      devShells.default =
        let
          pkgs = import nixpkgs {
            inherit system;
            # bring devshell attribute into the pkgs
            overlays = [ devshell.overlays.default ];
          };
        in
        pkgs.devshell.mkShell {
          name = "hugo-devshell";
          # a list of packages to add to the shell environment.
          # Container image build/run (`just hugo build|push|run`) uses the host's docker or
          # podman CLI; the devshell provides none, so it cannot shadow the host's.
          packages = [
            pkgs.hugo
            pkgs.dart-sass
          ];
        };
    });
}
