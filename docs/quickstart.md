# Getting started

## Prerequisites

`hugo` requires `git` and `go`; it is recommended to install `dart-sass`.

For the sake of complication (and keeping a pristine local environment), I'll use `nix` with [numtide's `devshell`](https://numtide.github.io/devshell/getting_started.html) to install/manage the dependencies.

To start, run `nix develop` at the repo root.  Or if using `direnv`, add it to the `.envrc`

## Draft

```sh
hugo new content posts/<postname>.md
```

## Build

The static site can be generated with `hugo`, assuming the nix devshell is active

```sh
hugo -D # render drafts for local testing
```

or with the nix flake as in prod

```sh
nix build '.?submodules=1'
# nix outputs are in ./result
```

To run a local copy of the built site, run:

```sh
hugo server -c ./result/public
```

## Containerize

Once the site has been built, build the docker image:

```sh
# copy files from nix store to local dir
cp -r $(nix-store -qR result/)/public ./public
# build image
podman build \
  -t ghcr.io/ahgraber/aimlbling-about:$(date +"%Y.%m.%d-%H%M") \
  -t ghcr.io/ahgraber/aimlbling-about:latest \
  .
```
<!--
```sh
# build image with nix
nix build '.?submodules=1#blogImage'
# load it to podman/docker
podman load < result
``` -->

Run image

```sh
podman machine start
podman run --rm -p 8080:8080 ghcr.io/ahgraber/aimlbling-about:latest
```

### Nix & Numtide devshell

[Getting started - devshell](https://numtide.github.io/devshell/getting_started.html)
[Getting started with Nix Flakes and devshell](https://yuanwang.ca/posts/getting-started-with-flakes.html)

## Hugo

- [Getting started | Hugo](https://gohugo.io/getting-started/)
- [Hugo Markdown Reference | Markdown Guide](https://www.markdownguide.org/tools/hugo/)

### complications

- [Shortcodes | Hugo](https://gohugo.io/content-management/shortcodes/)
- [Footnote styling - support - HUGO](https://discourse.gohugo.io/t/footnote-styling/114/6)

## Themes

- [Hextra | Hugo Themes](https://themes.gohugo.io/themes/hextra/)
<!-- - [Coder | Hugo Themes](https://themes.gohugo.io/themes/hugo-coder/) - supports giscus for comments; umami for analytics
- [Typo | Hugo Themes](https://themes.gohugo.io/themes/typo/) -->

## [giscus](https://giscus.app/)

Watch [changelog](https://github.com/giscus/giscus/blob/main/CHANGELOG.md) for pseudo-versioning

- configure postgres (note *T* in POSTGRES*T*_...) -(POSTGREST_URL, POSTGREST_ROLE, POSTGREST_SECRET)

Notes/References:

- `giscus.json` should be placed in the root of the repository whose Discussions is used for giscus. It's used to control how giscus can interact with your repository.
- [Giscus do not show when self-hosting · giscus · Discussion #705](https://github.com/orgs/giscus/discussions/705)
- [giscus app refused to connect · Issue #380 · giscus/giscus](https://github.com/giscus/giscus/issues/380#issuecomment-1026328308)

## [Staticman](https://staticman.net/docs/)

- [Configuring Staticman Comments with Hugo - DEV Community](https://dev.to/julio_ui/configuring-staticman-comments-with-hugo-317g)
- [Staticman - A comments system for Hugo site - support - HUGO](https://discourse.gohugo.io/t/staticman-a-comments-system-for-hugo-site/4429/7)
- [Configuring Staticman Comments with Hugo - DEV Community](https://dev.to/julio_ui/configuring-staticman-comments-with-hugo-317g)
- - [Running Staticman on Hugo Blog With Nested Comments - Yasoob Khalid](https://yasoob.me/posts/running_staticman_on_static_hugo_blog_with_nested_comments/#troubleshooting)
- [Adding Staticman Comments | Performance Matters](https://travisdowns.github.io/blog/2020/02/05/now-with-comments.html#integrate-comments-into-site)
