"""Data retrieval module for fetching data from local or remote sources.

This module provides functionality to retrieve data from either local storage
or remote URLs, with caching support.
"""

from http import HTTPStatus
from pathlib import Path

import pandas as pd
import requests


class DataGetter:
    """Handles data retrieval from local or remote locations.

    This class manages data fetching with support for both local file system
    and remote HTTP sources, with optional refresh/caching behavior.
    """

    def __init__(
        self, local_data_location: Path, remote_data_location: str, refresh: bool
    ):
        """Initialize the DataGetter.

        Args:
            local_data_location: Path to local data storage directory.
            remote_data_location: Base URL for remote data source.
            refresh: If True, always fetch fresh data; if False, use cached data.
        """
        self.local_data_location = local_data_location
        self.remote_data_location = remote_data_location
        self.refresh = refresh

    def get_data_path(self, data_name: str) -> str | Path:
        """Get data path from either remote or local location.

        Assumes all data files are CSV pandas dataframes.
        First checks for local file, then tries remote URL.

        Args:
            data_name: Name of the data file to retrieve.

        Returns:
            str | Path: Path to local file or URL to remote resource.

        Raises:
            FileNotFoundError: If data not found locally or remotely.
            RuntimeError: If data retrieval fails due to HTTP errors.
        """
        local_path = self.local_data_location / data_name

        if local_path.exists():
            return local_path

        remote_path = f"{self.remote_data_location.rstrip('/')}/{data_name}"

        response = requests.get(remote_path, timeout=30)

        if response.status_code == HTTPStatus.OK:
            return remote_path

        if response.status_code == HTTPStatus.NOT_FOUND:
            msg = f"Could not find {data_name} on the internet ({remote_path}) or locally at {local_path}"
            raise FileNotFoundError(msg)

        # Handle other HTTP errors
        msg = f"Failed to retrieve {data_name} from {remote_path} (status: {response.status_code})"
        raise RuntimeError(msg)

    def get_generic_data(self, data_location: str, output_file_name: Path):
        """Get data from a datasource and store it in the output file location.

        The output location is expected to be in the output folder.
        Will use cached data if available unless refresh is True.

        Args:
            data_location: Name or path of the data source.
            output_file_name: Path where to save the pickled dataframe.
        """
        if output_file_name.exists() and not self.refresh:
            return

        data_path = self.get_data_path(data_location)

        pd.read_csv(data_path).to_pickle(output_file_name)
