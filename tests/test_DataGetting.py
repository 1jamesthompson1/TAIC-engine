import os

import pytest

import engine.gather.DataGetting as DataGetting


@pytest.mark.parametrize(
    "file_name, expected",
    [
        pytest.param("event_types.csv", True, id="event_types.csv"),
        pytest.param("non-existent_file.csv", False, id="Failed attempt"),
    ],
)
def test_get_generic(tmpdir, file_name, expected):
    output_file = tmpdir.join("test_data.pkl")

    dataGetter = DataGetting.DataGetter(
        "data",
        "https://raw.githubusercontent.com/1jamesthompson1/TAIC-engine/main/data/",
        False,
    )

    if expected:
        dataGetter.get_generic_data(file_name, output_file)

        assert os.path.exists(output_file)

    else:
        with pytest.raises(FileNotFoundError):
            dataGetter.get_generic_data(file_name, output_file)
