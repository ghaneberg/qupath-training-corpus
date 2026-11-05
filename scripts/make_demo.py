#!/usr/bin/env python3
"""Thin dev wrapper around ``python -m qupath_corpus.demo``.

Kept so the repo works when run straight from a source checkout without
installing: adds src/ to sys.path then delegates to the demo.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qupath_corpus.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
