"""Configuration file management utilities."""

from pathlib import Path

import yaml


class ConfigFile:
    """Base class for managing configuration files."""

    def __init__(self, file_path: Path = Path("config.yaml")):
        """Initialize configuration file manager.

        Args:
            file_path: Path to the configuration file. Defaults to "config.yaml".

        Raises:
            ValueError: If the configuration file does not exist.
        """
        self._file_path = file_path

        if not self._file_exists():
            msg = f"{file_path} does not exist"
            raise ValueError(msg)

    def _file_exists(self) -> bool:
        return self._file_path.exists()


class ConfigReader(ConfigFile):
    """Read configuration from YAML files."""

    def __init__(self, file_path: Path = Path("config.yaml")):
        """Initialize configuration reader.

        Args:
            file_path: Path to the configuration file. Defaults to "config.yaml".
        """
        super().__init__(file_path)
        self._config = self._read_config()

    def _read_config(self) -> dict:
        with self._file_path.open("r") as f:
            return yaml.safe_load(f)

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            The loaded configuration as a dictionary.
        """
        return self._config


class ConfigWriter(ConfigFile):
    """Write configuration to YAML files."""

    def __init__(self):
        """Initialize configuration writer."""
        super().__init__()

    def write_config(self, config) -> None:
        """Write configuration to file.

        Args:
            config: Configuration dictionary to write.
        """
        with self._file_path.open("w") as f:
            yaml.dump(config, f)


config_reader = ConfigReader()
config_writer = ConfigWriter()
