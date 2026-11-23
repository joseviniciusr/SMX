"""Pytest plugin local to ``tests/``: options, markers and shared fixtures.

Pytest loads this file automatically before collecting the tests of this
folder; no test imports it.  It does *not* decide which tests run (that is
test discovery plus ``-k`` / ``-m`` selection); it only *provides*:

* the ``--update-golden`` command-line option (see ``test_regression.py``);
* session-scoped fixtures (built once for the whole run).  Fitted objects are
  shared, so tests must treat them as **read-only** — use ``copy.deepcopy``
  before mutating anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``helpers`` lives next to this file; make it importable as a plain module
# regardless of the directory pytest is launched from.
sys.path.insert(0, str(Path(__file__).parent))

from helpers import LinearSoftmaxStub, fit_smx, make_known_truth  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/data/golden_global.json from the current code "
             "(only when a change of the global SMX output is intentional).",
    )


@pytest.fixture(scope="session")
def known_truth():
    """Scenario S1 (one marker zone per class + a noise zone) and its stub."""
    data = make_known_truth(seed=7)
    data["est"] = LinearSoftmaxStub(n_ch=data["n_ch"], n_zones=data["n_zones"])
    return data


@pytest.fixture(scope="session")
def fitted_global(known_truth):
    return fit_smx(known_truth, known_truth["est"])


@pytest.fixture(scope="session")
def fitted_within(known_truth):
    return fit_smx(known_truth, known_truth["est"], per_class="within")


@pytest.fixture(scope="session")
def fitted_all(known_truth):
    return fit_smx(known_truth, known_truth["est"], per_class="all")
