# Releasing the Chatbot

How to cut a release of `sme-chatbot` and pull the resulting image. Driven by
[`.github/workflows/release-chatbot.yml`](../../.github/workflows/release-chatbot.yml),
triggered by pushing a tag.

---

## How it works

Pushing a tag matching `chatbotv2/vMAJOR.MINOR.PATCH[-prerelease]` (e.g.
`chatbotv2/v0.3.0`, `chatbotv2/v0.3.0-alpha.1`) runs the **Release Chatbot**
workflow, which:

1. Validates the tag and parses the version.
2. Runs the test gate (`tests/unit`, `tests/backend_api`).
3. Builds the Docker image for `linux/amd64` and `linux/arm64`, each on its
   own runner (arm64 under QEMU, ~20 min), and pushes each untagged to GHCR
   (GitHub Container Registry) at `ghcr.io/b-arol-o/chatbotv2`.
4. Once both platforms are in, tags them as one multi-platform image with
   the version. Non-prerelease tags additionally get `:latest`. If either
   platform fails, nothing is tagged and no Release is created.
5. Creates a GitHub Release on the tag with auto-generated notes.

The pipeline authenticates its own push with the ephemeral `GITHUB_TOKEN`
(scoped via `permissions: packages: write` in the workflow) — no repo secret
is involved in publishing.

---

## Where the version lives

There is **no version file to bump**. The tag is the version, and the release
build stamps it into the image in three places:

| Where | How to read it | Set by |
| --- | --- | --- |
| `GET /health` → `version`, and OpenAPI `info.version` | `curl -s http://<host>:8000/health` | `APP_VERSION` build arg → env var → `config.APP_VERSION` |
| OCI image labels | `docker inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' <image>` | `docker build --label` in the release workflow's Build image step |
| Image tag itself | `ghcr.io/b-arol-o/chatbotv2:0.3.0` | the pushed git tag |

A build with no `APP_VERSION` build arg — any local `docker build`, or running
the app from a checkout — reports `dev`. That is the intended signal for "not
a release build", not a bug.

The one place that does **not** carry the release version is the Python
distribution metadata: `pyproject.toml` pins `version = "0.0.0"` as a fixed
placeholder, so `pip show sme-chatbot` and
`importlib.metadata.version("sme-chatbot")` say `0.0.0` in every image. That
is deliberate — nothing reads it at runtime, and `0.0.0` is
recognisably a placeholder rather than a stale real version. **Do not bump it
at release time**; use `/health` or the OCI labels to identify a deployment.

---

## Pre-flight checks

Before tagging:

- The default branch (the branch releases are tagged from) is green —
  **CI Chatbot** passing on the exact commit you're releasing (it runs on
  `push` to that branch, not just on PRs; the workflow triggers on `main` and
  `dev-v2`, so the default branch must be one of those two):

  ```bash
  BRANCH=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name)
  gh run list --workflow ci-chatbot.yml --branch "$BRANCH" --limit 1 \
    --json headSha,conclusion,status
  ```

  Check the reported `headSha` matches `git rev-parse "origin/$BRANCH"`. If there
  is **no run** for that commit, it is because the merge touched none of the
  workflow's path filter (`src/sme_chatbot/**`, `frontends/**`,
  `tests/unit/**`, `tests/backend_api/**`, `pyproject.toml`, `uv.lock`,
  `Dockerfile`, the workflow itself) — a docs-only or scripts-only commit
  cannot move the test gate, so an absent run there is expected, not a
  missing signal. For anything else, an absent run means the release gate
  has not been exercised: don't tag until one appears.
- `uv run pytest tests/unit tests/backend_api` passes locally — the same gate
  the release workflow re-runs, so a local failure will fail the release too.
- Decide the version. Check the latest tag: `git tag -l 'chatbotv2/v*' --sort=-v:refname | head -5`.
  The workflow warns (but doesn't block) if the new
  version isn't higher than the last one.
- Pre-releases (`-alpha.N`, `-rc.N`) never get `:latest` — only a stable
  `MAJOR.MINOR.PATCH` tag does.

---

## Cutting a release

Tag and push (no version file to bump — the tag *is* the version; see
[Where the version lives](#where-the-version-lives)):

```bash
git tag chatbotv2/v0.3.0
git push origin chatbotv2/v0.3.0
```

Or create the tag via the GitHub Releases UI on `github.com`, which pushes it
and triggers the workflow the same way.

**The push is what releases, not the tag.** `git tag` alone is a purely local
ref: nothing runs, no image is built, no Release appears. A tag created and
left unpushed looks identical to a released one in `git tag -l`, so confirm it
reached the remote before assuming the release is under way:

```bash
git ls-remote --tags origin | grep chatbotv2/v0.3.0
```

Watch the run under **Actions → Release Chatbot**. On success:

- A GitHub Release exists at the tag, with auto-generated notes.
- The image is at `ghcr.io/b-arol-o/chatbotv2:0.3.0`
  (and `:latest` if it wasn't a pre-release).

---

## Pulling the image

The registry is **GHCR** (`ghcr.io`, GitHub's own container registry). The
package is public, so it pulls anonymously, with no login and no token:

```bash
docker pull ghcr.io/b-arol-o/chatbotv2:0.3.0
# or the floating stable tag:
docker pull ghcr.io/b-arol-o/chatbotv2:latest
```

### If the pull fails with `unauthorized`

An anonymous pull is refused in two cases:

- **The package is not public (yet).** A GHCR package has its own visibility,
  separate from the repository's: a newly published package starts private
  and stays so when the repository is made public. An owner of the
  organization changes it once, on the package page under **Package settings
  → Change visibility**. Until then — and in any fork that keeps its package
  private — pull with a credential, as below.
- **A stale credential is stored for `ghcr.io`.** An expired token in the
  Docker config is rejected even for a public image. Run
  `docker logout ghcr.io` and pull again.

To pull a private package, the Docker CLI needs its own credential (being
signed in to `github.com` in a browser does not help):

1. Create a personal access token (classic) on `github.com` with the
   `read:packages` scope: `https://github.com/settings/tokens`.
2. Log in once:

   ```bash
   echo "<your-PAT>" | docker login ghcr.io \
     -u <your-github-username> --password-stdin
   ```

3. Pull as above.

If the pull is still `unauthorized` after logging in, your GitHub account
lacks read access to the package itself (package permissions are managed
separately from repository permissions, under the package's **Manage access**
settings) — ask an owner of the package to grant it.

---

## Rolling back

There's no delete/unpublish step — roll back by pointing consumers at the
previous version's tag:

```bash
docker pull ghcr.io/b-arol-o/chatbotv2:<previous-version>
```

To move the floating `:latest` tag back to a previous stable release, re-run
the release workflow's tag/push logic manually against that version, or ask
whoever operates the deployment to pin the compose/deployment config to the
explicit previous version tag instead of `:latest`.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `pip show sme-chatbot` in the image says `0.0.0` | Expected — the Python metadata version is a fixed placeholder; read the release from `GET /health` or the OCI labels, see [Where the version lives](#where-the-version-lives) |
| `GET /health` reports `"version": "dev"` | The image was built without the `APP_VERSION` build arg — i.e. it is not a release-workflow build |
| Tag push doesn't trigger the workflow | Tag doesn't match `chatbotv2/vMAJOR.MINOR.PATCH[-prerelease]` — check the regex in the workflow's "Validate tag & parse version" step |
| Tag exists locally but there's no Release and no workflow run | The tag was never pushed — `git tag -l` shows local tags too. Check with `git ls-remote --tags origin`, then `git push origin <tag>` |
| `docker pull` → `unauthorized` | No `docker login` done yet, or your PAT lacks `read:packages` — see [Pulling the image](#pulling-the-image) |
| `docker pull` → `unauthorized` even after login | Your account lacks read access to the package — ask a repo admin to grant it via the package's "Manage Actions access" settings |
| Push step fails in CI | GHCR (Packages) may not be enabled for the repo/org — see the archive fallback commented in the workflow (attach the image to the Release instead) |
