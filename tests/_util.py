"""
Shared test helpers for the cucc test suite
"""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(REPO_ROOT, "bin")
INCLUDE_DIR = os.path.join(REPO_ROOT, "include")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

CUCC = os.path.join(BIN_DIR, "cucc")

def load_transpiler():
    # import bin/transpile.py as a module regardless of current directory
    path = os.path.join(BIN_DIR, "transpile.py")
    spec = importlib.util.spec_from_file_location("cucc_transpile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def heavy_enabled():
    # check if resource-intensive tasks should run
    return os.environ.get("CUCC_RUN_HEAVY", "").strip() not in ("", "0", "false")

if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)
