# [AI/MLbling About](https://aimlbling-about.ninerealmlabs.com)

In which I organize my thoughts about LLMs, Generative AI, cool papers from arxiv.org, and other marginalia on topics in data science and machine learning.

## Quickstart

This blog is built with [Hugo](https://gohugo.io/) using the [Hextra](https://imfing.github.io/hextra/) theme.
`hugo` and its prerequisites `git`, `go` and `dart-sass` are required.

I also use [Task](https://taskfile.dev/) to define standard tasks (see ./Taskfile.yaml and ./.taskfiles/hugo/taskfile.yaml),
such as automatically creating a new branch per post for drafting (`task: hugo:new title="<title>"`),
and publishing posts when ready (`task hugo:publish path="<relative path to index.md>"`).

## Drafting new posts

Drafts are managed in their own `blog/<title>` branch.

Content is written using "standard" markdown ([Goldmark](https://github.com/yuin/goldmark), specifically).
The [Hextra](https://imfing.github.io/hextra/) theme provides additional features, like shortcodes for
[callouts](https://imfing.github.io/hextra/docs/guide/shortcodes/callout/), [cards](https://imfing.github.io/hextra/docs/guide/shortcodes/cards/), [hide/show details](https://imfing.github.io/hextra/docs/guide/shortcodes/details/), etc.
Footnotes are supported, as is [LaTeX](https://imfing.github.io/hextra/docs/guide/latex/), [Mermaid diagrams](https://imfing.github.io/hextra/docs/guide/diagrams/), and syntax highlighting.

Once a draft is finished, it can be published (`task hugo:publish path="<relative path to index.md>"`) on its branch,
then PR'd into `main`.

## Deployment

Once the pull request is complete, a Github Action will build a new multi-arch container image and push to `ghcr.io/ahgraber/aimlbling-about`.
Versioning is datetime-based (`YYYY.MM.DDHHMM`) since semver doesn't really make sense (no sense of major.minor.patch for a blog).

## Hosting

It is hosted on my [kubernetes (k3s) homelab](https://github.com/ahgraber/homelab-gitops-k3s).
A [Renovate](https://docs.renovatebot.com/) workflow in my homelab repo watches for new image versions.
Once detected, it will automatically create a PR to deploy the new blog image to prod.

## Integrations

[ ] [giscus](https://giscus.app/) - a comments system powered by GitHub Discussions
[x] [umami analytics](https://umami.is/) - a simple, fast, **privacy-focused** alternative to Google Analytics
