from unet import UNet

import torch
import unittest


class TestUNet(unittest.TestCase):
    def test_basic(self):
        model = UNet()
        B, C, H, W = 10, 3, 128, 128
        t = torch.randint(0, 100, (B, 1))
        x = torch.rand(B, C, H, W)
        out = model(x, t)

        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
