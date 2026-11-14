# Releasing rvfitter

Versions come from git tags (`hatch-vcs`): tag `v2.1.0` builds as `2.1.0`.
Untagged builds get a `.devN+g<hash>` version; `3.0.0.dev0` is the fallback when there is no git metadata.

## One-time setup (trusted publishing, no API tokens)

1. On <https://pypi.org/manage/account/publishing/> add a pending publisher: project `rvfitter`, owner `Will-Cooper`, repository `rvfitter`, workflow `release.yml`, environment `pypi`.
2. Repeat on <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi` (optionally add yourself as a required reviewer on `pypi`).

## Cutting a release

1. Update `version` and `date-released` in `CITATION.cff`; commit to `main`.
2. Check that CI is green and that `ruff check src tests` is clean.
3. Optional dry run: Actions > Release > Run workflow (publishes the build to TestPyPI; a version can be uploaded only once).
4. Tag, then publish a GitHub release for the tag: `git tag v2.1.0 && git push origin v2.1.0`.
   Publishing the release triggers the `Release` workflow, which builds, runs `twine check` and uploads to PyPI.
5. Check `pip install "rvfitter[all]"` in a fresh environment and `rvfitter doctor`.

The HITRAN-derived telluric model and the template grids are never bundled; users build or download them (see `install.md`).
