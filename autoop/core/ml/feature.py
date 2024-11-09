from typing import Literal


class Feature:
    def __init__(self, name: str,
                 type: Literal["numerical", "categorical"]) -> None:
        """
        Initializes a Feature object.

        Args:
            name (str): The name of the feature.
            type (Literal["numerical", "categorical"]):
                The type of the feature, either "numerical" or "categorical".
        """
        self.name = name
        self.type = type

    def __str__(self):
        """
        Returns a string representation of the feature object.

        Returns:
            str: A string in the format "name: type" where 'name'
            is the feature's name and 'type' is the feature's type.
        """
        return f"{self.name}: {self.type}"
