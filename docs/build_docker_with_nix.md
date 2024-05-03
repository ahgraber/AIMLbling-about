# Building docker images with nix

## References

- [Nix is a better Docker image builder than Docker's image builder - Xe Iaso](https://xeiaso.net/talks/2024/nix-docker-build/)
- [Rust Environment and Docker Build with Nix Flakes | John's Codes](https://johns.codes/blog/rust-enviorment-and-docker-build-with-nix-flakes#make-a-docker-image)
- [Using Nix with Dockerfiles – Mitchell Hashimoto](https://mitchellh.com/writing/nix-with-dockerfiles)
- [Building on dockerfile-based images - Help - NixOS Discourse](https://discourse.nixos.org/t/building-on-dockerfile-based-images/29583/11)

- [jnsgr.uk/flake.nix at main · jnsgruk/jnsgr.uk](https://github.com/jnsgruk/jnsgr.uk/blob/main/flake.nix)

## Old (broken) flake for reference

```nix
{
  description = "A simple flake to install Hugo test environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.devshell.url = "github:numtide/devshell";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  # inputs.giscus.url = "github:giscus/giscus";

  outputs = { self, devshell, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          devshell.overlays.default
        ];
      };

      version = "${self.sourceInfo.lastModifiedDate}";

      blog = pkgs.stdenv.mkDerivation {
        name = "AIMLbling About";
        src = ./.;
        buildInputs = with pkgs; [
          hugo
          dart-sass
        ];
        buildPhase = ''
          hugo
        '';
        installPhase = ''
          mkdir -p $out
          cp -r public $out/
        '';
      };

      # giscusBuild = {};

    in
    {
      packages = {
        inherit blog;
        # blog = blogBuild;

        blogImage = pkgs.dockerTools.buildLayeredImage {
          name = "ghcr.io/ahgraber/aimlbling-about";
          tag = version;
          # fromImage = pkgs.dockerTools.pullImage parsedImageMeta;
          fromImage = pkgs.dockerTools.pullImage {
            # NOTE get metadata with `nix-prefetch-docker <registry>/<org>/<imagename>:<tag>`
            imageName = "ghcr.io/nginxinc/nginx-unprivileged";
            imageDigest = "sha256:ef60d54ac69279e59fda0495d52611b74f9f223970ce94ab3e8b4cad166f5a50";
            sha256 = "189jav5b0c5paixihkn8pdv02bhlk5qhbjk7awjwdrm70q0ia25g";
            finalImageName = "ghcr.io/nginxinc/nginx-unprivileged";
            finalImageTag = "latest";
          };
          contents = [
            blog
          ];
          # TODO: copy from /public
          # extraCommands = ''
          #   cp ./public /usr/share/nginx/html
          # '';
          config = {
            # Cmd = [ "nginx" "-g" "daemon off;" ];
            # WorkingDir = "/app";
            Expose = 8080;
          };
        };
      };
      defaultPackage = blog;

      devShell = pkgs.devshell.mkShell {
        name = "hugo-devshell";
        # a list of packages to add as build inputs
        # buildInputs = with pkgs; [
        #   nodejs
        # ];
        # a list of packages to add to the shell environment
        packages = with pkgs; [
          hugo
          dart-sass
          nodejs
          nix-prefetch-docker
          podman
        ];
        # imports = [ (pkgs.devshell.importTOML ./devshell.toml) ];
      };
    }
  );
}
```
