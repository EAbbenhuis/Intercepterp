"""Pytest bootstrap.

Inserts the repository root at the front of sys.path so that test modules and
the packages under test (envs, training, eval, viz) import cleanly regardless of
where pytest is invoked from. This file's mere presence at the repo root also
fixes the rootdir for pytest.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
