import argparse
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from fbias.cli import train
from fbias.logging import get_logger, set_level

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction

registry: dict[str, Callable[[_SubParsersAction[ArgumentParser]], ArgumentParser]] = {
    "train": train.add_cmd,
}

logger = get_logger(__name__)

__all__ = ["main", "registry"]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fbias",
        description="On frequency bias of TSFMs.",
        epilog="Run `fbias <command> -h` for a command's options.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="emit debug logging"
    )
    subparsers = parser.add_subparsers(dest="cmd", metavar="<command>")
    for register in registry.values():
        register(subparsers)

    args = parser.parse_args()
    if args.verbose:
        set_level(logging.DEBUG)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
