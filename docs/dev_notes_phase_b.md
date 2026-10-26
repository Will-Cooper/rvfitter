# Phase B notes

- `rvfitter-legacy` is a conda clone of the old `rvfitter` env (Python 3.9, astropy 5.0.4, specutils 1.7.0, pytest 8.4.2). Legacy tests: `rvfitter-legacy/bin/python -m pytest tests/legacy` (47 pass). The J1745 harness works with `--code-dir <worktree>/legacy` and still gives 28.95 ± 5.45 km/s.
- `tests/legacy/test_a6_side_effects.py` also needed its `ROOT` pointed at `legacy/`.
- The updated `rvfitter` env (in-place `conda env update --prune`) has Python 3.12.15, astropy 8.0.2 and specutils 2.4.0.
- Ruff runs clean with the pyupgrade rules on. The type modules use built-in generics (`list`, `dict`, `X | None`).
- The default pytest run ignores `tests/legacy`; the `data` and `gui` markers are registered.

## The rvfitter env is an editable install of the development worktree

During development, the `rvfitter` env's editable install points at
`~/Documents/rvfitter/.claude/worktrees/v2-modernise`, not at the main checkout.
That worktree is locked, so it is not removed automatically. Keep it (choose "keep" if asked when the session ends) until the branch is merged.
After the `v2-modernise` PR is merged and `~/Documents/rvfitter` is on the merged branch, repoint the env:

```bash
conda activate rvfitter
pip install -e "$HOME/Documents/rvfitter[all]"
python -c "import rvfitter, pathlib; print(pathlib.Path(rvfitter.__file__).parent)"   # should print ~/Documents/rvfitter/src/rvfitter
```

Only then remove the worktree (`git worktree remove .claude/worktrees/v2-modernise`).
The old v1 code keeps running in `rvfitter-legacy` throughout.
