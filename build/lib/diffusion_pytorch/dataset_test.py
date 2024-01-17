from dataset import Dataset

import tempfile
import unittest


class TestDataset(unittest.TestCase):
    def test_creation(self):
        # `testdata` folder has one jpg file.
        dataset = Dataset(imgs_dir="./testdata")

        self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
