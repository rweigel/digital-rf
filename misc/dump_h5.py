#!/usr/bin/env python3
"""Recursively find and dump the contents of all HDF5 files in a directory."""

import sys
import os
import argparse
import h5py
import numpy as np


def dump_item(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] /{name}")
        for key, val in obj.attrs.items():
            print(f"{indent}  @{key}: {val!r}")
    elif isinstance(obj, h5py.Dataset):
        shape_str = str(obj.shape)
        dtype_str = str(obj.dtype)
        print(f"{indent}[Dataset] /{name}  shape={shape_str}  dtype={dtype_str}")
        for key, val in obj.attrs.items():
            print(f"{indent}  @{key}: {val!r}")
        # Print data (truncate large arrays)
        data = obj[()]
        if data.size <= 100:
            print(f"{indent}  data: {data!r}")
        else:
            flat = data.flat
            preview = [next(flat) for _ in range(10)]
            print(f"{indent}  data (first 10 of {data.size}): {preview!r} ...")


def dump_h5_file(path):
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    try:
        with h5py.File(path, "r") as f:
            # Root attributes
            for key, val in f.attrs.items():
                print(f"  @{key}: {val!r}")
            f.visititems(dump_item)
    except Exception as exc:
        print(f"  ERROR reading file: {exc}")


def find_h5_files(directory):
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith(".h5") or fname.endswith(".hdf5") or fname.endswith(".he5"):
                yield os.path.join(root, fname)


def main():
    parser = argparse.ArgumentParser(
        description="Dump contents of all HDF5 files under a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: {directory!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    found = False
    for h5path in find_h5_files(directory):
        found = True
        dump_h5_file(h5path)

    if not found:
        print(f"No HDF5 files found under {directory}")


if __name__ == "__main__":
    main()
