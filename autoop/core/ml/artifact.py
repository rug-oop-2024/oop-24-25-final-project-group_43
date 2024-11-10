from pydantic import BaseModel, Field, validator
import base64
from typing import Literal, Optional
import pandas as pd
import pickle
from abc import ABC, abstractmethod


class Artifact(BaseModel, ABC):
    """
    Artifact class representing a machine learning artifact.
    """
    type: str
    name: str
    asset_path: str
    version: str
    tags: Optional[list[str]] = None
    data: Optional[bytes]
    metadata: Optional[dict] = None

    @property
    def id(self) -> str:
        """
        Generates a unique identifier for the artifact by encoding the
        asset path in base64 and appending the version number.

        Returns:
            str: The unique identifier for the artifact,
                with base64 encoded asset path
                and version number, where '=' characters are
                replaced with '_'.
        """
        # Encode the asset_path in base64
        path = base64.b64encode(self.asset_path.
                                encode('ascii')).decode('ascii')
        full_path = f"{path}_{self.version}".replace("=", "_")
        print(full_path)
        return full_path

    def save(self, data: bytes) -> bytes:
        """
        Save the given data to the file specified by asset_path.

        Args:
            data (bytes): The data to be written to the file.

        Returns:
            bytes: The data that was written to the file.
        """
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)

    def read(self) -> bytes:
        """
        Reads the data stored in the artifact.

        Returns:
            bytes: The data stored in the artifact.
        """
        return self.data


    @staticmethod
    def from_pipeline(cls,
                      type: str,
                      name: str,
                      asset_path: str,
                      version: str,
                      tags,
                      data: object,
                      metadata) -> 'Artifact':
        """
        Create an Artifact instance from a pipeline.

        Args:
            type (str): The type of the artifact.
            name (str): The name of the artifact.
            asset_path (str): The path to the asset.
            version (str): The version of the artifact.
            tags: Tags associated with the artifact.
            data (object): The data to be serialized and stored in the artifact.
            metadata: Additional metadata for the artifact.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        return cls(
            type=type,
            name=name,
            asset_path=asset_path,
            version=version,
            tags=tags,
            data=pickle.dumps(data),
            metadata=metadata)

    def to_pipeline(self) -> object:
        """
        Deserializes the stored data into a pipeline object.

        Returns:
            object: The deserialized pipeline object.
        """
        return pickle.loads(self.data)
