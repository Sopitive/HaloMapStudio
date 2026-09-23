# Halo Map Studio — documentation site

This folder is the user documentation for Halo Map Studio, published with GitHub Pages.

## Format

Plain, hand-written **HTML + one CSS file**, no build step, no JavaScript framework, no
external fonts or CDNs. `.nojekyll` tells GitHub Pages to serve the files exactly as they are
(no Jekyll processing). This was chosen over Jekyll/Markdown because it needs no theme, no
`_config.yml`, no Ruby toolchain, renders identically when opened from disk, and cannot break
when GitHub changes its default Jekyll theme.

## Enabling GitHub Pages for this folder

1. Push this repository to GitHub (the `site/` folder must be on the branch you publish from,
   normally `main`).
2. In the repository go to **Settings ▸ Pages**.
3. Under **Build and deployment** choose **Source: Deploy from a branch**.
4. Set **Branch** to `main` (or your default branch) and **Folder** to **`/site`**… GitHub
   only offers `/ (root)` and `/docs` in that drop-down, so either:
   * rename this folder to `docs/` and pick **`/docs`**, or
   * keep `site/` and publish it with the **GitHub Actions** source instead: choose
     **Source: GitHub Actions**, add a workflow that uploads `site/` with
     `actions/upload-pages-artifact` (`path: site`) and deploys it with
     `actions/deploy-pages`. A minimal workflow:

     ```yaml
     name: Pages
     on:
       push:
         branches: [main]
         paths: ["site/**"]
     permissions:
       pages: write
       id-token: write
     jobs:
       deploy:
         runs-on: ubuntu-latest
         environment:
           name: github-pages
         steps:
           - uses: actions/checkout@v4
           - uses: actions/upload-pages-artifact@v3
             with:
               path: site
           - id: deployment
             uses: actions/deploy-pages@v4
     ```
5. Save. The site appears at `https://<owner>.github.io/<repo>/` within a minute or two.

All links inside the site are relative, so it works at any base path (project page or a
custom domain) and when opened locally by double-clicking `index.html`.

## Editing

* Every page is a self-contained `.html` file with the navigation repeated at the top; when
  you add a page, add it to the `<nav>` list in each file (or regenerate with the helper
  script that produced these files).
* `style.css` holds all styling, including the dark-mode palette.
* Screenshots live in `assets/` as 1280×720 JPEGs (quality 85). They were rendered with the
  tool's own headless mode, for example
  `HMS_MAP=forge_halo HMS_CAM="-80,120,110,-60,-25" HMS_SHOT=forge-world.png HMS_SHOT_SIZE=1280x720 ./hms-app`.
* Keep the total size of `assets/` well under 15 MB.

## Pages

| File | Content |
|---|---|
| `index.html` | Overview, screenshots, download link, licence summary |
| `getting-started.html` | Requirements, install, first launch, settings location |
| `maps-and-variants.html` | Map discovery and groups, `.mvar` open/save, MCC save paths |
| `editing.html` | Selection, placement, transforms, Object panel, mass edit, undo, CAD tools |
| `camera.html` | Flying, framing, stand-off, screenshots |
| `shortcuts.html` | Every keyboard/mouse shortcut |
| `rendering.html` | Settings window and renderer accuracy |
| `scripting.html` | Complete Forge script reference and the MCP server |
| `headless.html` | Headless rendering, batch scripts, CLI flags, environment variables |
| `halo4.html` | Halo 4 status |
| `faq.html` | Troubleshooting |
| `license.html` | Licence summary and commercial use |
| `changelog.html` | Release notes |
