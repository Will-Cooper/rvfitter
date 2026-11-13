"""Command-line interface: init, run, measure, inject, forecast, templates, gui and doctor.

Exit codes: 0 clean, 1 usage, config or runtime error, 2 the run finished but a file was skipped or a method failed.
"""
import argparse
import importlib
import logging
import sys
import urllib.request
from pathlib import Path

from . import __version__, logging_setup
from .pipeline import NoMethodsError

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
SVO_URL = 'https://svo2.cab.inta-csic.es/theory/newov2/index.php'
ZENODO_URL = 'https://zenodo.org'
NET_TIMEOUT = 3  # s


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
    p.add_argument('--telluric-corrected', action='store_true',
                   help='the spectra are already telluric-corrected (relaxes the telluric masking)')
    p.add_argument('--no-plots', action='store_true', help='do not make diagnostic plots')
    p.add_argument('--bary', choices=('auto', 'apply', 'skip'), default=None, help='barycentric correction mode')

    p = sub.add_parser('run', parents=[common], help='run an analysis described by a TOML config')
    p.add_argument('config', help='TOML configuration file')
    p.add_argument('--no-plots', action='store_true', help='do not make diagnostic plots')
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


def _finish(res, failures: list[tuple[str, str]], out: Path, formats, cfg) -> int:
    """Log and write the results, and return the exit code (2 if anything failed; `pipeline.run` logs which)"""
    from . import plotting, results

    if not res:
        log.error('No objects were measured')
        return 1
    for line in results.summary_lines(res):
        log.info(line)
    results.write(res, out, formats, config=cfg)
    if cfg.output.plots:
        plotting.save_all(res, out.parent)
    return 2 if failures else 0


def _cmd_measure(args) -> int:
    from . import config, pipeline

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
    if args.telluric_corrected:
        cfg.input.telluric_corrected = True
    if args.no_plots:
        cfg.output.plots = False
    methods = [m for m in args.methods.split(',') if m] if args.methods else None
    failures: list[tuple[str, str]] = []
    res = pipeline.run(cfg, args.path, n_procs=args.n_procs, medium=args.medium, methods=methods, bary=args.bary,
                       failures=failures)
    return _finish(res, failures, out, args.formats, cfg)


def _cmd_run(args) -> int:
    from . import config, pipeline

    cfg = config.load(args.config)
    if args.no_plots:
        cfg.output.plots = False
    inputs = cfg.input.paths
    if not inputs:
        log.error('The config %s names no input paths', args.config)
        return 1
    out = Path(cfg.output.path or 'results.csv')
    logging_setup.configure(_verbosity(args), out.parent / 'rvfitter.log')
    failures: list[tuple[str, str]] = []
    res = pipeline.run(cfg, inputs, n_procs=args.n_procs, failures=failures)
    return _finish(res, failures, out, cfg.output.formats or None, cfg)


def _cmd_init(args) -> int:
    from . import config, io

    spec = io.read_spectrum(Path(args.spectrum))
    cfg = config.default_for(spec)
    config.dump(cfg, Path(args.output))
    log.info('Wrote default config to %s', args.output)
    return 0


def _reachable(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=NET_TIMEOUT):  # noqa: S310
            return True, url
    except OSError as exc:  # URLError and timeouts
        return False, f'{url}: {type(exc).__name__}'


def _doctor_checks(offline: bool) -> list[tuple[str, bool, str, str]]:
    """(name, ok, detail, fix hint) for each check"""
    checks = []

    def add(name, ok, detail='', hint=''):
        checks.append((name, bool(ok), detail, hint))

    v = sys.version_info
    add('Python >= 3.11', v >= (3, 11), f'{v.major}.{v.minor}.{v.micro}',
        'conda create -n rvfitter -c conda-forge python=3.12')
    for mod, extra in (('numpy', None), ('scipy', None), ('astropy', None), ('specutils', None), ('pandas', None),
                       ('matplotlib', None), ('PySide6', 'gui'), ('emcee', 'mcmc'), ('corner', 'mcmc')):
        hint = f'pip install "rvfitter[{extra}]"' if extra else 'pip install --force-reinstall rvfitter'
        try:
            m = importlib.import_module(mod)
            add(f'import {mod}', True, getattr(m, '__version__', ''), hint)
        except (ImportError, OSError) as exc:
            add(f'import {mod}', False, f'{type(exc).__name__}: {exc}'.splitlines()[0], hint)
    try:
        QtCore = importlib.import_module('PySide6.QtCore')
        plugdir = Path(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms'
        plats = sorted(p.stem.removeprefix('libq') for p in plugdir.glob('*q*')) if plugdir.is_dir() else []
        add('Qt platform plugin', bool(plats), ', '.join(plats) or 'none found',
            'headless machine: set QT_QPA_PLATFORM=offscreen; otherwise reinstall PySide6')
    except (ImportError, OSError) as exc:
        add('Qt platform plugin', False, f'PySide6 unavailable ({type(exc).__name__})', 'pip install "rvfitter[gui]"')
    from . import templates
    cache = templates.cache_dir()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / '.doctor'
        probe.write_text('ok')
        probe.unlink()
        add('Template cache writable', True, str(cache), '')
    except OSError as exc:
        add('Template cache writable', False, f'{cache}: {exc}', 'fix its permissions, or pass --dir to templates')
    for name, url in (('SVO reachable', SVO_URL), ('Zenodo reachable', ZENODO_URL)):
        if offline:
            add(name, True, 'skipped (--offline)')
        else:
            ok, detail = _reachable(url)
            add(name, ok, detail, 'check the proxy (HTTPS_PROXY), or use --offline and a local template directory')
    return checks


def _cmd_doctor(args) -> int:
    logging_setup.configure(_verbosity(args))
    checks = _doctor_checks(args.offline)
    width = max(len(c[0]) for c in checks)
    for name, ok, detail, hint in checks:
        log.info('%s  %s  %s%s', 'OK  ' if ok else 'FAIL', name.ljust(width), detail, '' if ok else f'  -> fix: {hint}')
    bad = [c[0] for c in checks if not c[1]]
    if bad:
        log.error('Failed: %s', ', '.join(bad))
        return 1
    log.info('All checks passed')
    return 0


_EXTRAS = {'PySide6': 'gui', 'emcee': 'mcmc', 'corner': 'mcmc'}


def _import_failure(exc: BaseException, command: str) -> int:
    """Log an ImportError and return 1; an absent rvfitter module is reported as not yet available"""
    name = getattr(exc, 'name', None) or ''
    root = name.split('.')[0]
    if root in _EXTRAS:
        log.error('rvfitter %s needs %s: install it with  pip install "rvfitter[%s]"', command, root, _EXTRAS[root])
        return 1
    if root == 'rvfitter' or isinstance(exc, (AttributeError, NotImplementedError)):
        log.error('rvfitter %s is not yet available', command)
        return 1
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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in LAZY:  # parse their own options (argparse's REMAINDER rejects leading ones)
        common = ('-v', '--verbose', '-q', '--quiet')
        args = argparse.Namespace(command=argv[0], rest=[a for a in argv[1:] if a not in common],
                                  verbose=any(a in common[:2] for a in argv[1:]),
                                  quiet=any(a in common[2:] for a in argv[1:]))
    else:
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:  # argparse: 0 for --help/--version, 2 for a usage error
            return 1 if exc.code else 0
        if args.command is None:
            parser.print_help()
            return 1
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
    except NoMethodsError as exc:
        logging_setup.configure(_verbosity(args))
        log.error('%s', exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logging_setup.configure(_verbosity(args))
        log.error('%s: %s', type(exc).__name__, exc)
        log.debug('traceback', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
