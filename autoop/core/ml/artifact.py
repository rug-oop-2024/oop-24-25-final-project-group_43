from pydantic import BaseModel, Field, validator
import base64
from typing import Literal, Optional
import pandas as pd
import pickle
from abc import ABC, abstractmethod


class Artifact(BaseModel, ABC):
    type: str
    name: str
    asset_path: str
    version: str
    tags: Optional[list[str]] = None
    data: Optional[bytes]
    metadata: Optional[dict] = None

    @property
    def id(self) -> str:
        # Encode the asset_path in base64
        path = base64.b64encode(self.asset_path.
                                encode('ascii')).decode('ascii')
        full_path = f"{path}_{self.version}".replace("=", "_")
        print(full_path)
        return full_path

    def save(self, data: bytes) -> bytes:
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)

    def read(self) -> bytes:
        return self.data

    def get(self, param: str) -> str:
        if param == "type":
            return self.type

    def from_pipeline(cls, type: str, name: str, asset_path: str, version: str, tags, data: object, metadata) -> 'Artifact':
        return cls(
            type=type,
            name=name,
            asset_path=asset_path,
            version = version,
            tags=tags,
            data=pickle.dumps(data),
            metadata=metadata)

    def to_pipeline(self) -> object:
        return pickle.loads(self.data)

    # def save_pipeline_artifact(self, name: str, version: str, ) -> None:
    #     with open(f'pipelines/{name}_{version}.pkl', 'wb') as file:
    #         file.write(self.data)
