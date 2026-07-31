"""Shared pytest fixtures for the Crop Recommendation System test suite."""

from __future__ import annotations

import matplotlib
import pytest

# A non-interactive backend, chosen here so it is in force before any test
# module imports pyplot: the suite must run on machines with no display
# attached (CI included).
matplotlib.use("Agg")

from src.data import RAW_DATA_PATH, load_data  # noqa: E402

#: Skip marker applied to tests that need the real, committed CSV.
requires_raw_dataset = pytest.mark.skipif(
    not RAW_DATA_PATH.is_file(),
    reason=(
        f"Raw dataset missing at {RAW_DATA_PATH}. It is meant to be committed to the "
        "repository; restore it before running the data tests."
    ),
)


@pytest.fixture(scope="session")
def raw_data():
    """Load the raw dataset once per test session, without validating it.

    Validation is switched off here so that the validation tests can make their
    own assertions rather than inheriting an exception from the fixture.
    """
    return load_data(validate=False)
