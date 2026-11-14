# Releasing rvfitter

Versions come from git tags (`hatch-vcs`): tag `v2.1.0` builds as `2.1.0`.
Untagged builds get a `.devN+g<hash>` version; `2.1.0.dev0` is the fallback when there is no git metadata.

## What happens on a release
- **Publishing a GitHub release** runs `bundles.yml`. It builds the standalone GUI apps (Windows, macOS on Apple Silicon, Linux) with PyInstaller, smoke-tests the Linux one, and attaches the three archives to the release.
  - Zenodo archives the release and mints a version DOI under the concept DOI 10.5281/zenodo.6624927.
- **PyPI is separate and manual:** Actions > "Publish to PyPI" > Run workflow, choosing `testpypi` or `pypi`. A version can be uploaded only once.

## One-time PyPI setup (trusted publishing, no API tokens)
1. On <https://pypi.org/manage/account/publishing/> add a pending publisher: project `rvfitter`, owner `Will-Cooper`, repository `rvfitter`, workflow `release.yml`, environment `pypi`.
2. Repeat on <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi` (optionally add yourself as a required reviewer on `pypi`).

## Cutting a release
1. Update `version` and `date-released` in `CITATION.cff`, and commit.
2. Check that CI is green, including the bundle builds on the pull request.
3. Create the release: `gh release create v2.1.0 --target main --title v2.1.0 --generate-notes` (or tag and publish it on GitHub). The bundles attach themselves within about 15 minutes.
4. Optionally publish to PyPI as above, then check `pip install "rvfitter[all]"` in a fresh environment and run `rvfitter doctor`.

The HITRAN-derived telluric model and the template grids are never bundled; users build or download them (see `install.md`).
