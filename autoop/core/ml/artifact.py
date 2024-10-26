from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):
    def __init__(self, name: str, data: bytes):
        name = name
        version: str
        asset_path: str
        tags: list
        metadata: str
        type: str
        data = data
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
        return data

    @property
    def read(self):
        return self.data

    @property
    def get(self, type: str):
        return self.data
    
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