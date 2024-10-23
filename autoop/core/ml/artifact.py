from pydantic import BaseModel, Field
import base64

class Artifact(BaseModel):

    name: str
    version: str
    asset_path: str
    tags: list
    metadata: str
    type: str
    data: list
    _id: str

    @property
    def id(self, asset_path: str, version: str) -> str:
        _id = asset_path.encode('ascii')
        _id = base64.b64encode(_id)
        _id = _id  + ':' + version
        return _id
    