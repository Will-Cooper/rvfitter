"""Command-line interface: init, run, measure, inject, forecast, templates, gui and doctor.

Exit codes: 0 ok, 1 runtime error, 2 usage error (including subcommands that are not yet available).
"""
import argparse
import importlib
import logging
import shutil
import sys
import urllib.request
from pathlib import Path

from . import __version__, logging_setup

__all__ = ['main', 'build_parser']

log = logging.getLogger(__name__)

SUBCOMMANDS = ('init', 'run', 'measure', 'inject', 'forecast', 'templates', 'gui', 'doctor')
# Lazily dispatched subcommands: command -> (module, function, pass remaining argv?)
LAZY = {
    'inject': ('rvfitter.inject', 'main', True),
    'forecast': ('rvfitter.forecast', 'main', True),
    'templates': ('rvfitter.templates', 'cli_main', True),
    'gui': ('rvfitter.gui.app', 'main', False),
}
SVO_URL = 'https://svo2.cab.unige.ch/theory/newov2/index.php'


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-v', '--verbose', action='store_true', help='show DEBUG messages on the console')
    common.add_argument('-q', '--quiet', action='store_true', help='show only warnings and errors')

    parser = argparse.ArgumentParser(prog='rvfitter', description='Radial velocities of brown dwarfs.')
    parser.add_argument('--version', action='version', version=f'rvfitter {__version__}')
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('measure', parents=[common], help='measure RVs of a spectrum, directory or archive')
    p.add_argument('path', help='spectrum file, directory, .zip or tarball')
    p.add_argument('--config', help='TOML configuration file')
    p.add_argument('--methods', help='comma-separated list of methods (default: from the config)')
    p.add_argument('--out', default='results.csv', help='output table (default: results.csv)')
    p.add_argument('--format', dest='formats', help='comma-separated output formats: csv,json,tsv,html')
    p.add_argument('--n-procs', type=int, default=None, help='processes used within each object')
    p.add_argument('--medium', choices=('auto', 'air', 'vacuum'), default=None, help='wavelength medium of the input')
    p.add_argument('--bary', choices=('auto', 'apply', 'skip'), default=None, help='barycentric correction mode')

    p = sub.add_parser('run', parents=[common], help='run an analysis described by a TOML config')
    p.add_argument('config', help='TOML configuration file')
    p.add_argument('--n-procs', type=int, default=None, help='processes per object (default: from the config)')

    p = sub.add_parser('init', parents=[common], help='write a default config for a spectrum')
    p.add_argument('spectrum', help='spectrum file the defaults are derived from')
    p.add_argument('-o', '--output', default='rvfitter.toml', help='config file to write (default: rvfitter.toml)')

    p = sub.add_parser('doctor', parents=[common], help='check the installation')
    p.add_argument('--offline', action='store_true', help='skip the network check')

    for name in ('inject', 'forecast', 'templates', 'gui'):
        p = sub.add_parser(name, parents=[common], help=f'{name} (optional component)', add_help=False)
        p.add_argument('rest', nargs=argparse.REMAINDER)
    return parser


def _verbosity(args) -> int:
    return 1 if getattr(args, 'verbose', False) else -1 if getattr(args, 'quiet', False) else 0


def _cmd_measure(args) -> int:
    from . import config, pipeline, results

    out = Path(args.out)
    logging_setup.configure(_verbosity(args), out.parent / 'rvfitter.log')
    if args.config:
        cfg = config.load(args.config)
    else:
        from . import archives, io
        first = archives.collect_inputs(Path(args.path))
        if not first:
            log.error('No spectra found in %s', args.path)
            return 1
        cfg = config.default_for(io.read_spectrum(first[0], medium=args.medium or 'auto'))
    methods = [m for m in args.methods.split(',') if m] if args.methods else None
    res = pipeline.run(cfg, args.path, n_procs=args.n_procs, medium=args.medium, methods=methods, bary=args.bary)
    if not res:
        log.error('No objects were measured')
        return 1
    for line in results.summary_lines(res):
        log.info(line)
    results.write(res, out, args.formats, config=cfg, dump=config.dump)
    return 0


def _cmd_run(args) -> int:
    from . import config, pipeline, results

    cfg = config.load(args.config)
    inputs = cfg.input.paths
    if not inputs:
        log.error('The config %s names no input paths', args.config)
        return 2
    out = Path(cfg.output.path or 'results.csv')
    logging_setup.configure(_verbosity(args), out.parent / 'rvfitter.log')
    res = pipeline.run(cfg, inputs, n_procs=args.n_procs)
    if not res:
        log.error('No objects were measured')
        return 1
    for line in results.summary_lines(res):
        log.info(line)
    results.write(res, out, cfg.output.formats or None, config=cfg, dump=config.dump)
    return 0


def _cmd_init(args) -> int:
    from . import config, io

    spec = io.read_spectrum(Path(args.spectrum))
    cfg = config.default_for(spec)
    config.dump(cfg, Path(args.output))
    log.info('Wrote default config to %s', args.output)
    return 0


def _doctor_checks(offline: bool) -> list[tuple[str, bool, str, bool]]:
    """(name, ok, detail, required) for each check"""
    checks = []

    def add(name, ok, detail='', required=True):
        checks.append((name, bool(ok), detail, required))

    add('Python >= 3.11', sys.version_info >= (3, 11), sys.version.split()[0])
    for mod, req in (('numpy', True), ('scipy', True), ('astropy', True), ('specutils', True), ('pandas', True),
                     ('matplotlib', True), ('PySide6', False), ('emcee', False)):
        try:
            m = importlib.import_module(mod)
            add(f'import {mod}', True, getattr(m, '__version__', ''), req)
        except Exception as exc:  # noqa: BLE001
            add(f'import {mod}', False, f'{type(exc).__name__}: {exc}'.splitlines()[0], req)
    try:
        import os
        QtCore = importlib.import_module('PySide6.QtCore')
        plugdir = Path(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms'
        plats = sorted(p.name for p in plugdir.glob('libq*')) if plugdir.is_dir() else []
        add('Qt platform plugin', bool(plats), f"{os.environ.get('QT_QPA_PLATFORM', 'default')}; {len(plats)} found",
            False)
    except Exception as exc:  # noqa: BLE001
        add('Qt platform plugin', False, f'PySide6 unavailable ({type(exc).__name__})', False)
    add('LaTeX (plot labels)', shutil.which('latex') is not None, shutil.which('latex') or 'not on PATH', False)
    try:
        import platformdirs
        cache = Path(platformdirs.user_cache_dir('rvfitter'))
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / '.doctor'
        probe.write_text('ok')
        probe.unlink()
        add('Template cache writable', True, str(cache))
    except Exception as exc:  # noqa: BLE001
        add('Template cache writable', False, str(exc))
    if offline:
        add('SVO reachable', True, 'skipped (--offline)', False)
    else:
        try:
            with urllib.request.urlopen(SVO_URL, timeout=3):  # noqa: S310
                add('SVO reachable', True, SVO_URL, False)
        except Exception as exc:  # noqa: BLE001
            add('SVO reachable', False, f'{type(exc).__name__}', False)
    return checks


def _cmd_doctor(args) -> int:
    logging_setup.configure(_verbosity(args))
    checks = _doctor_checks(args.offline)
    width = max(len(c[0]) for c in checks)
    for name, ok, detail, required in checks:
        mark = '✓' if ok else '✗'
        note = '' if ok or required else ' (optional)'
        log.info('%s %s  %s%s', mark, name.ljust(width), detail, note)
    bad = [c[0] for c in checks if not c[1] and c[3]]
    if bad:
        log.error('Required checks failed: %s', ', '.join(bad))
        return 1
    log.info('All required checks passed')
    return 0


_EXTRAS = {'PySide6': 'gui', 'emcee': 'mcmc', 'corner': 'mcmc'}


def _import_failure(exc: BaseException, command: str) -> int:
    """Log an ImportError: a missing optional dependency is a runtime error (1), an absent rvfitter module is
    "not yet available" (2)"""
    name = getattr(exc, 'name', None) or ''
    root = name.split('.')[0]
    if root in _EXTRAS:
        log.error('rvfitter %s needs %s: install it with  pip install "rvfitter[%s]"', command, root, _EXTRAS[root])
        return 1
    if root == 'rvfitter' or isinstance(exc, (AttributeError, NotImplementedError)):
        log.error('rvfitter %s is not yet available', command)
        return 2
    log.error('rvfitter %s: missing dependency %s', command, name or exc)
    return 1


def _cmd_lazy(args) -> int:
    modname, func, takes_argv = LAZY[args.command]
    logging_setup.configure(_verbosity(args))
    try:
        fn = getattr(importlib.import_module(modname), func)
    except (ImportError, AttributeError) as exc:
        return _import_failure(exc, args.command)
    rest = [a for a in args.rest if a != '--']
    try:
        out = fn(rest) if takes_argv else fn()
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return int(out or 0)


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help()
        return 2
    handlers = {'measure': _cmd_measure, 'run': _cmd_run, 'init': _cmd_init, 'doctor': _cmd_doctor}
    handler = handlers.get(args.command, _cmd_lazy)
    try:
        return handler(args)
    except KeyboardInterrupt:
        log.error('Interrupted')
        return 1
    except (ImportError, NotImplementedError) as exc:
        logging_setup.configure(_verbosity(args))
        return _import_failure(exc, args.command)
    except Exception as exc:  # noqa: BLE001
        logging_setup.configure(_verbosity(args))
        log.error('%s: %s', type(exc).__name__, exc)
        log.debug('traceback', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
