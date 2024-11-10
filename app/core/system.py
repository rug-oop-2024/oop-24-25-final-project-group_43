from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    A class to manage the registration, retrieval, listing, and deletion
    of artifacts.
    Attributes:
        database (Database): The database instance to store artifact metadata.
        storage (Storage): The storage instance to save artifact data.
    Methods:
        register(artifact: Artifact):
            Registers a new artifact by saving its data in storage and its
            metadata in the database.
        list(type: str = None) -> List[Artifact]:
            Lists all artifacts, optionally filtered by type.
        get(artifact_id: str) -> Artifact:
            Retrieves an artifact by its ID.
        delete(artifact_id: str):
            Deletes an artifact by its ID, removing both its data from storage
            and its metadata from the database.
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes the System class with a database and storage.

        Args:
            database (Database): The database instance to be used by
                the system.
            storage (Storage): The storage instance to be
                used by the system.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving its data to storage
            and its metadata to the database.

        Args:
            artifact (Artifact): The artifact to be registered, containing
                data, asset path, name, version, tags, metadata, and type.

        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        List artifacts from the database, optionally filtered by type.
        Args:
            type (str, optional): The type of artifacts to filter by.
                Defaults to None.
        Returns:
            List[Artifact]: A list of Artifact objects.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve an artifact from the database and storage.

        Args:
            artifact_id (str): The unique identifier of the
                artifact to retrieve.

        Returns:
            Artifact: An instance of the Artifact class.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact from both storage and database.

        This method retrieves the artifact data from the database using the
            provided artifact ID, deletes the associated asset from storage,
            and then removes the artifact entry from the database.

        Args:
            artifact_id (str): The unique identifier of the artifact
                to be deleted.

        Raises:
            KeyError: If the artifact with the given ID
                does not exist in the database.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    AutoMLSystem is a singleton class that manages the storage and database
        for an automated machine learning system.

    Attributes:
        _instance (AutoMLSystem): The singleton instance of the AutoMLSystem.
        _storage (LocalStorage): The storage system used to store artifacts.
        _database (Database): The database used to manage metadata.
        _registry (ArtifactRegistry): The registry that manages artifacts
            in the system.

    Methods:
        __init__(storage: LocalStorage, database: Database):
            Initializes the AutoMLSystem with the given
                storage and database.

        get_instance() -> AutoMLSystem:
            Returns the singleton instance of the AutoMLSystem.
                If the instance does not exist, it creates one.

        registry() -> ArtifactRegistry:
            Returns the artifact registry associated with the AutoMLSystem.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoMLSystem with the given storage and database.

        Args:
            storage (LocalStorage): The local storage instance to be used.
            database (Database): The database instance to be used.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieves the singleton instance of the AutoMLSystem class.
            If the instance does not exist, it initializes it with
            LocalStorage and Database objects.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem class.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Returns the registry.

        Returns:
            dict: The registry artifact registry.
        """
        return self._registry
