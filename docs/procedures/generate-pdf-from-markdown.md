# HOWTO Generate an A4 PDF from a Markdown file

## Overview

This project provides two methods to render Markdown files to A4-sized PDF, using
[pandoc](https://pandoc.org/) with the [WeasyPrint](https://weasyprint.org/) PDF engine.

### Method 1: GitHub Action (Recommended for CI/CD)

The [`.github/workflows/publish-md-pdf.yml`](../../.github/workflows/publish-md-pdf.yml) GitHub Action
automatically converts Markdown files to PDF on pull requests or on-demand. This is the recommended
approach for CI/CD pipelines and automated doc publishing.

**For pull requests:** Any `.md` file in the `docs/` folder that is modified will be automatically
converted to PDF and uploaded as a workflow artifact.

**For on-demand conversion:** Use the workflow dispatch trigger to manually specify which Markdown
files to convert:
- Go to the **Actions** tab in GitHub
- Select **Publish Markdown as PDF**
- Click **Run workflow**
- Enter space-separated file paths (e.g., `docs/procedures/file1.md docs/procedures/file2.md`)
- PDFs will be generated and available as artifacts

### Method 2: Local Shell Script (Docker wrapper)

`scripts/publish-md-pdf.sh` is a wrapper around the upstream Docker image
([B-AROL-O/publish-md-pdf](https://github.com/B-AROL-O/publish-md-pdf)) and renders
one or more Markdown files to A4-sized PDF on your local machine. By default, each
`<file>.md` is written as `<file>.pdf` into the current working directory.

This is useful for local development when you want to preview how a Markdown doc will look as a PDF.
The script automatically handles Docker volume mounts, so you don't need to worry about paths.

## Prerequisites

### For the local shell script (`Method 2`)

Docker must be installed and running. The script uses the upstream Docker image
[ghcr.io/b-arol-o/publish-md-pdf](https://github.com/B-AROL-O/publish-md-pdf) and automatically
handles all dependencies (pandoc, weasyprint, etc.) inside the container.

Install Docker for your OS: [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

### For the GitHub Action (`Method 1`)

No prerequisites — the action runs in the GitHub Actions environment and handles dependencies
automatically.

## Usage

### GitHub Action (Method 1)

The workflow is triggered automatically when you push changes to Markdown files in the `docs/` folder.
To manually trigger a conversion:

1. Go to the **Actions** tab in GitHub
2. Select **Publish Markdown as PDF**
3. Click **Run workflow**
4. Enter the Markdown file paths (space-separated)
5. Monitor the run, then download the generated PDFs from the **Artifacts** section

### Local Shell Script (Method 2)

```bash
scripts/publish-md-pdf.sh [--output-dir DIR] [--output-name NAME] [--css-file FILE] <file.md> [file2.md ...]
```

From the repository root, to render the Langfuse setup procedure:

```bash
scripts/publish-md-pdf.sh docs/procedures/langfuse-setup.md
```

This produces `./langfuse-setup.pdf` in the current working directory.

Multiple files can be passed in one invocation:

```bash
scripts/publish-md-pdf.sh docs/procedures/langfuse-setup.md docs/procedures/docker-deployment.md
```

### Options

- `--output-dir DIR` — directory to write the PDF(s) into (default: `$PWD`). Created if it doesn't
  already exist.
- `--output-name NAME` — filename for the generated PDF (default: derived from the input `.md`
  filename). Only valid when converting a single input file, since multiple inputs would otherwise
  collide on the same output name. `.pdf` is appended automatically if not already present.
- `--css-file FILE` — stylesheet to use (default: `scripts/publish-md-pdf.css`). A custom stylesheet
  should keep the `.task-checkbox` rules from the default one (see [Notes](#notes) below) if the
  source Markdown has GFM task lists (`- [ ]` / `- [x]`).

```bash
scripts/publish-md-pdf.sh --output-dir dist --output-name langfuse-setup \
  docs/procedures/langfuse-setup.md
```

## Notes

- Page layout (A4 size, margins, page-number footer, code-block wrapping) is built into the
  upstream Docker image and cannot be customized locally.
- Task-list checkboxes (`- [ ]` / `- [x]`) are rendered correctly with proper styling.
- If the Markdown file has a YAML front matter `title:` field, it is used as the PDF's document
  title metadata.
- Generated PDFs inherit the sensitivity of their source Markdown — check the source before sharing
  the PDF outside its intended context. PDF outputs are not tracked by git; add them to `.gitignore`
  or write them under an ignored directory (e.g. `dist/`) if you generate them in-tree.

## When to use each method

| Method | When to use |
| --------------------------------- | ------------------------------------------------------------ |
| **GitHub Action (Method 1)** | Automated CI/CD workflows, PR reviews that need PDF previews |
| **Local Shell Script (Method 2)** | Local development, quick preview before commit, testing CSS |

<!-- EOF -->
