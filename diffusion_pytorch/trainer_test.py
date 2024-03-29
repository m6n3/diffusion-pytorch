from diffusion_pytorch import gaussian_diffusion as gd
from diffusion_pytorch import trainer as tr
from diffusion_pytorch import unet

import torch
import unittest


class TestTrainer(unittest.TestCase):
    def _fake_dataset(self, shape_chw=[3, 128, 128], length=2):
        class FakeDataset(torch.utils.data.Dataset):
            def __init__(self, shape_chw, length):
                super().__init__()
                self.shape = shape_chw
                self.length = length

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                return torch.rand(self.shape)

        return FakeDataset(shape_chw, length)

    def test_train(self):
        model = unet.UNet()
        diffusion = gd.GaussianDiffusion(model=model, max_timesteps=10)
        trainer = tr.Trainer(
            diffusion=diffusion,
            dataset=self._fake_dataset(),
            train_batch_size=2,
            save_folder="",
        )  # Do not save model.
        trainer.train()


if __name__ == "__main__":
    unittest.main()
