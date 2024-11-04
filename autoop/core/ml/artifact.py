from pydantic import BaseModel, Field, validator
import base64
from typing import Literal, Optional
import pandas as pd
import pickle
from abc import ABC,abstractmethod

class Artifact(BaseModel,ABC):
    type: str
    name: str
    asset_path: str
    version: str
    tags: Optional[list[str]] = None
    data: Optional[bytes]
    metadata: Optional[dict] = None
    _id: str


    @property
    def id(self, asset_path: str, version: str) -> str:
        _id = asset_path.encode('ascii')
        _id = base64.b64encode(_id)
        _id = _id.decode() + ':' + version
        return _id

    def save(self, data: bytes) -> bytes:
        with open(self.asset_path, 'wb') as file:
            file.write(self.data)


    def read(self):
        return self.data


    # @property
    # def get(self, param: str):
    #     if param == "type":
    #         return self.type
    

# with json you can convert the object to a string and then bytes and save it to a file
# data should be in bytes
# saved artifact should be in data

# model.dump
# What marco said about this class:
# baseclass for all the logical objects in the pipeline
# define an abstract method to dump the object to a file
# we want to have files that refers to this specific type of attr
# be able to save dataset, model, features
# find what the attr mean by checking the requirements and specific code
# very generic abstract class
# manage all types of artifacts