"""Classes for managing datasets in AutoOp."""
from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """A class that represents a Dataset, inherits from the Artifact class."""
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a new Dataset instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> 'Dataset':
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            name (str): The name of the dataset.
            asset_path (str): The path where the dataset assets are stored.
            version (str, optional): The version of the dataset.
            Defaults to "1.0.0".

        Returns:
            Dataset: An instance of the Dataset class.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads data from a source, decodes it from bytes to a string,
        and then loads it into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the data read from
            the source.
        """
        data_bytes = super().read()
        csv = data_bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the given DataFrame as a CSV file and return its
        bytes representation.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The bytes representation of the CSV file.
        """
        data_bytes = data.to_csv(index=False).encode()
        return super().save(data_bytes)
