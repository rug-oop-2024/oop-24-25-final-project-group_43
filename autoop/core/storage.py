from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found.

    Attributes:

    Methods:
        __init__(path: str): Initializes the exception with the given path.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the exception with the given path.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for storage operations.

    This class defines the interface for storage operations,
    including saving, loading, deleting, and listing data. Subclasses
    must implement these methods to provide
    specific storage functionality.

    Methods:
        save(data: bytes, path: str) -> None:
            Save data to a given path.

        load(path: str) -> bytes:
            Load data from a given path.

        delete(path: str) -> None:
            Delete data at a given path.

        list(path: str) -> list:
            List all paths under a given path.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    LocalStorage is a class that provides.

    Attributes:
        _base_path (str): The base directory path where assets will be stored.

    Methods:
        __init__(base_path: str = "./assets") -> None:

        save(data: bytes, key: str) -> None:

        load(key: str) -> bytes:

        delete(key: str = "/") -> None:

        list(prefix: str = "/") -> List[str]:

        _assert_path_exists(path: str) -> None:
            Check if the specified path exists and raise an error if it
            does not.

        _join_path(path: str) -> str:
            Join the base path with the given path and normalize it.
    """
    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the storage with a specified base path.

        Args:
            base_path (str): The base directory path where assets
            will be stored. Defaults to "./assets".

        Raises:
            OSError: If the directory cannot be created.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save the given data to a file specified by the key.

        This method ensures that the parent directories of the file path
        are created if they do not already exist.

        Args:
            data (bytes): The data to be saved.
            key (str): The key used to determine the file path where
                the data will be saved.

        Returns:
            None
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load the contents of a file as bytes.

        Args:
            key (str): The key representing the file path to load.

        Returns:
            bytes: The contents of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete the file or directory at the specified key.

        Args:
            key (str): The key representing the path to the file or
            directory to delete. Defaults to "/".

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files in the storage with the given prefix.

        Args:
            prefix (str): The prefix path to list files from. Defaults to "/".

        Returns:
            List[str]: A list of file paths relative to the base path.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in
                keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
