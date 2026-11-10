#!/usr/bin/env python3
"""Prepare the Base3 RNA-FM cache without starting model training."""

import argparse

from rnafm_cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATAFILE,
    DEFAULT_MODEL_PATH,
    prepare_rnafm_cache,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datafile', default=str(DEFAULT_DATAFILE))
    parser.add_argument('--cache-dir', default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        '--model-path',
        default=None,
        help='default search: RNAFM_MODEL_PATH, {}, then the torch hub cache'.format(
            DEFAULT_MODEL_PATH
        ),
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default=None)
    parser.add_argument('--rebuild', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    prepare_rnafm_cache(
        datafile=args.datafile,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
        device=args.device,
    )


if __name__ == '__main__':
    main()
