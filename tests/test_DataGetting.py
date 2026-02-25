"""Tests for DataGetting module."""

from pathlib import Path

import pytest

from engine import DataGetting


@pytest.mark.parametrize(
    "file_name, expected",
    [
        pytest.param("event_types.csv", True, id="event_types.csv"),
        pytest.param("non-existent_file.csv", False, id="Failed attempt"),
    ],
)
def test_get_generic(tmp_path, file_name, expected):
    """Test generic data fetching functionality."""
    output_file = tmp_path / "test_data.pkl"

    data_getter = DataGetting.DataGetter(
        Path("data"),
        "https://raw.githubusercontent.com/1jamesthompson1/TAIC-engine/main/data/",
        False,
    )

    if expected:
        data_getter.get_generic_data(file_name, output_file)

        assert output_file.exists()

    else:
        with pytest.raises(FileNotFoundError):
            data_getter.get_generic_data(file_name, output_file)
