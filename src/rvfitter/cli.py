"""Command-line interface: init, run, measure, inject, forecast, templates, gui and doctor."""
import argparse
import sys

from . import __version__

SUBCOMMANDS = ('init', 'run', 'measure', 'inject', 'forecast', 'templates', 'gui', 'doctor')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rvfitter', description='Radial velocities of brown dwarfs.')
    parser.add_argument('--version', action='version', version=f'rvfitter {__version__}')
    sub = parser.add_subparsers(dest='command')
    for name in SUBCOMMANDS:
        sub.add_parser(name, help=f'{name} (not yet implemented)')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    print(f'rvfitter {args.command}: not yet implemented')
    return 2


if __name__ == '__main__':
    sys.exit(main())
