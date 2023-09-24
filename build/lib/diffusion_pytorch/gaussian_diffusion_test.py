from gaussian_diffusion import GaussianDiffusion
from unet import UNet

import torch
import unittest


class TestGaussinDiffusion(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = UNet()
        self.diffusion = GaussianDiffusion(model=self.model, max_timesteps=500)

    def test_forward(self):
        B, C, H, W = 10, 3, 128, 128
        x = torch.rand([B, C, H, W])
        loss = self.diffusion(x)

        self.assertTrue(0 <= loss)

    def test_sample(self):
        B, C, H, W = 1, 3, 128, 128
        imgs = self.diffusion.sample(batch_size=B, timesteps=1, shape_chw=[C, H, W])

        self.assertEqual(imgs.shape, torch.Size([B, C, H, W]))


if __name__ == "__main__":
    unittest.main()
