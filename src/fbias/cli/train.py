from __future__ import annotations

from fbias.logging import get_logger
from fbias.model import SimTFM

logger = get_logger(__name__)


def main(args):
    return 0


def add_cmd(subparsers):
    p = subparsers.add_parser(
        "train",
        help="Next patch prediction training.",
        description=(
            "Train a TFM model object through next patch prediction over synthetic "
            "data with controlled frequency distribution and noise level."
        ),
    )
    p.set_defaults(func=main)
    return p
