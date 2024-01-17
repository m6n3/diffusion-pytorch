from diffusion_pytorch import dataset as ds

import tempfile
import unittest


class TestDataset(unittest.TestCase):
    def test_creation(self):
        # `testdata` folder has one jpg file.
        dataset = ds.Dataset(imgs_dir="diffusion_pytorch/testdata")

        self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
