# Deployment

- GHA build container to homelab host?
  - figure out GHA action for clean docker build
  - figure out auto tag strategy (datetime?) for versioning
    - branch merge and release rules
  - k8s: need to autoapprove blog update PRs with renovate
  - refs:
    - <https://github.com/Euvaz/BlogHugo>
    - <https://github.com/Euvaz/GitOps-Home/blob/main/kubernetes/apps/professional/bloghugo/app/helmrelease.yaml>
- ~GHA build to github pages (w/ custom domain?)~
- GHA build to cloudflare pages?
  - [cloudflare/pages-action](https://github.com/cloudflare/pages-action)
  - [Deploy a Hugo site · Cloudflare Pages docs](https://developers.cloudflare.com/pages/framework-guides/deploy-a-hugo-site/)
  - [Redirecting *.pages.dev to a Custom Domain · Cloudflare Pages docs](https://developers.cloudflare.com/pages/how-to/redirect-to-custom-domain/)

## TODOs

- add non-disqus comments capability
  - [Staticman: Static sites with superpowers](https://staticman.net/)
    - [docker deployment](https://github.com/eduardoboucas/staticman/blob/master/docs/docker.md)
    - [Adding Staticman Comments | Performance Matters](https://travisdowns.github.io/blog/2020/02/05/now-with-comments.html#integrate-comments-into-site)
    - [Running Staticman on Hugo Blog With Nested Comments - Yasoob Khalid](https://yasoob.me/posts/running_staticman_on_static_hugo_blog_with_nested_comments/#step-5-adding-comment-partials-in-hugo-theme)
  - [utterances](https://utteranc.es/)

## Themes

- [Complete List | Hugo Themes](https://themes.gohugo.io/)
- [onweru/hugo-swift-theme: A simple open source theme for publishing with hugo](https://github.com/onweru/hugo-swift-theme?tab=readme-ov-file)
- [Partials · Blowfish](https://blowfish.page/docs/partials/)

## References

- [The proper way to build a Hugo docker container · Vasko's Blog](https://sko.ai/blog/how-to-actually-build-hugo-containers/)
- k8s-at-home
  - [rwaltr/rwaltpro: Source code for website](https://github.com/rwaltr/rwaltpro/tree/main)
  - [UnknownBlunders/website: a hugo blog at blog.blunders.dev](https://github.com/UnknownBlunders/website/tree/main)
  - [Euvaz/BlogHugo: Personal Hugo blog utilizing Docker and CI/CD via GitHub Actions.](https://github.com/Euvaz/BlogHugo)
  - [erenfro/BlogHugo: Personal Hugo blog utilizing Docker and CI/CD via GitHub Actions.](https://github.com/erenfro/BlogHugo/)
- [Nix is a better Docker image builder than Docker's image builder - Xe Iaso](https://xeiaso.net/talks/2024/nix-docker-build/)
