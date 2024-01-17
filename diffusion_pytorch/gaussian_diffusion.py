from diffusion_pytorch import noise_scheduler as ns

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(self, *, model, max_timesteps):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.model = model
        self.noise_scheduler = ns.NoiseScheduler(max_steps=max_timesteps + 1)

    def _denoise_and_add_noise(self, x, pred_noise, timesteps):
        # Do not add noise to the last step's image.
        new_noise = torch.rand_like(x) if timesteps > 1 else torch.zeros(x.shape)
        # new_noise: [batch_size, C, H, W]

        (
            pred_noise_coeff,
            new_noise_coeff,
            mean_x_coeff,
        ) = self.noise_scheduler.sampling_coeffs(timesteps)
        # pred_noise_coeff,new_noise_coeff,mean_x_coeff: [batch_size, 1, 1, 1]

        mean_x = x - pred_noise_coeff * pred_noise
        # mean_x: [batch_size, C, H, W]

        return mean_x_coeff * mean_x_coeff + new_noise_coeff * new_noise

    def get_model(self):
        return self.model

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def sample(self, batch_size, timesteps=None, shape_chw=[3, 128, 128]):
        """Generates `batch_size` number of images."""
        self.model.eval()

        x = torch.rand([batch_size, *shape_chw])
        timesteps = timesteps if timesteps is not None else self.max_timesteps

        for t in range(timesteps, 0, -1):
            batch_timesteps = torch.tensor([t] * batch_size).unsqueeze(-1)
            # timesteps: [batch_size, 1]

            pred_noise = self.model(x, batch_timesteps)
            # let C, H, W = *shape_chw
            # pred_noise : [batch_size, C, H, W]

            x = self._denoise_and_add_noise(x, pred_noise, t)
            # x: [batch_size, C, H, W]

        return x

    def forward(self, imgs):
        # imgs: [batch_size, C, H, W]

        batch_size = imgs.shape[0]

        timesteps = torch.randint(self.max_timesteps + 1, (batch_size, 1))
        # timesteps: [batch_size, 1]

        noisy_imgs, noises = self.noise_scheduler.noisify(imgs, timesteps=timesteps)
        # inputs_noisy, noise: [batch_size, C, H, W]

        pred_noises = self.model(noisy_imgs, timesteps)
        # pred_noises: [batch_size, C, H, W]

        loss = F.mse_loss(pred_noises, noises)
        # loss: [] scalar

        return loss
