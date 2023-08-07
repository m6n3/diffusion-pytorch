from dataset import Dataset

import tempfile
import unittest


class TestDataset(unittest.TestCase):
    def test_creation(self):
        # `testfiles` folder has one jpg file.
        dataset = Dataset(
            imgs_dir="./testfiles", transforms=Dataset.default_transforms()
        )

        self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
