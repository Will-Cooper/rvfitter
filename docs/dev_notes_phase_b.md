# Phase B notes

- `rvfitter-legacy` is a conda clone of the old `rvfitter` env (Python 3.9, astropy 5.0.4, specutils 1.7.0, pytest 8.4.2). Legacy tests: `rvfitter-legacy/bin/python -m pytest tests/legacy` (47 pass). The J1745 harness works with `--code-dir <worktree>/legacy` and still gives 28.95 +/- 5.45 km/s.
- `tests/legacy/test_a6_side_effects.py` also needed its `ROOT` pointed at `legacy/`.
- The updated `rvfitter` env (in-place `conda env update --prune`) has Python 3.12.15, astropy 8.0.2, specutils 2.4.0 and an editable install of this worktree. Note that the editable install points at the worktree path, so reinstall (`pip install -e ".[all]"`) after the branch is merged into the main checkout.
- Ruff: the `UP006/UP007/UP035/UP045` rules are ignored because the type-definition modules (not part of Phase B) use `typing.List/Optional`. `tests/synth.py:38` is 121 characters long and fails E501 (not part of Phase B); it needs wrapping by its owner.
- Default pytest run ignores `tests/legacy`; the `data` and `gui` markers are registered.
