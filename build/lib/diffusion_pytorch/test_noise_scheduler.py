""" Test for noise_scheduler.py """
from diffusion_pytorch import noise_scheduler as ns

import torch
import unittest


class TestNoiseScheduler(unittest.TestCase):
    def test_basic(self):
        B, C, H, W = 10, 3, 128, 128
        sample = torch.rand(B, C, H, W)
        noisy_sample, noise = ns.NoiseScheduler().noisify(
            sample, timesteps=torch.tensor([5]).unsqueeze(0)
        )

        self.assertEqual(sample.shape, noisy_sample.shape)
        self.assertEqual(sample.shape, noise.shape)
        self.assertFalse(torch.equal(sample, noisy_sample))


if __name__ == "__main__":
    unittest.main()
