from pydantic import BaseModel, Field, validator
import base64
from typing import Literal, Optional
import pandas as pd

class Artifact(BaseModel):

    #def __init__(self, type: str, name: str, asset_path: str, data: bytes, version: str):
    type: Literal['dataset', 'OneHotEncoder', 'StandardScaler']
    name: str
    asset_path: str
    version: str
    tags: Optional[list[str]] = None
    data: Optional[bytes]
    metadata: Optional[list[str]] = None
    _id: str


    @property
    def id(self, asset_path: str, version: str) -> str:
        _id = asset_path.encode('ascii')
        _id = base64.b64encode(_id)
        _id = _id  + ':' + version
        return _id

    # All the methods below are abstract and should be implemented in the subclasses
    # They are not done yet, just concepts of what should be according to pipeline.py
    @property
    def save(self, data: bytes) -> bytes:
        if not isinstance(data, bytes):
            raise TypeError(f"Data should be of type bytes, found {data.__class__.__name__}")
        return data

    @property
    def read(self):
        return self
        

    @property
    def get(self, param: str):
        if param == "type":
            return self.type
    

    def dump(self, data: bytes):
        return data

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