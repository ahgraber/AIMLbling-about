{
  description = "A simple flake to install Hugo test environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.devshell.url = "github:numtide/devshell";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, devshell, flake-utils, nixpkgs }:
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
          # imports = [];
          # a list of packages to add to the shell environment
          packages = [
            pkgs.hugo
            pkgs.dart-sass
            # pkgs.podman
            pkgs.colima
            pkgs.qemu
          ];
          # imports = [ (pkgs.devshell.importTOML ./devshell.toml) ];
        };
    });
}
