from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline

import unittest

# To make sure flakes8 doesn't complain about the unused import
TEST_DICT = {
    "test_database": TestDatabase,
    "test_storage": TestStorage,
    "test_features": TestFeatures,
    "test_pipeline": TestPipeline
}

if __name__ == '__main__':
    unittest.main()
