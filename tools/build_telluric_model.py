"""
Build the telluric transmission model from HITRAN (a thin wrapper around `rvfitter.calibration.telluric_build`).

Usage: python tools/build_telluric_model.py [OUT.npz [LINE_CACHE_DIR]]

The same as ``rvfitter templates telluric --build [--out OUT.npz] [--line-cache DIR]``. The default output is the
user cache (`rvfitter.calibration.telluric.cache_path()`). The model is built locally and must not be
redistributed (HITRAN states no redistribution licence); see `telluric_build` for the citation and terms.
"""
import logging
import sys

from rvfitter.calibration import telluric_build

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = sys.argv[1:]
    telluric_build.build(args[0] if args else None, line_cache=args[1] if len(args) > 1 else None, force=True)
