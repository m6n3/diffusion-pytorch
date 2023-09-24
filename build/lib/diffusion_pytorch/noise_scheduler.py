# forward.py
import torch


class NoiseScheduler:
    def __init__(self, beta_min=0.0001, beta_max=0.02, max_steps=500):
        self.betas = torch.linspace(beta_min, beta_max, max_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # \sqrt((1-b0)*(1-b_1)*(1-b_2)...(1-b_n))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # \sqrt(1-(1-b0)*(1-b_1)*(1-b_2)...(1-b_n))
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sampling_coeffs(self, timesteps):
        """Returns coeffiecients used in sampling."""
        pred_noise_coeff = (1 - self.alphas[timesteps, None, None]) / (
            1 - self.alphas_cumprod[timesteps, None, None]
        ).sqrt()
        new_noise_coeff = self.betas[timesteps, None, None].sqrt()
        sample_coeff = 1 / self.alphas[timesteps, None, None].sqrt()
        # pred_noise_coeff, new_noise_coeff, sample_coeff: [batch_size, 1, 1, 1,1]

        return pred_noise_coeff, new_noise_coeff, sample_coeff

    def noisify(self, sample, timesteps, noise=None, device="cpu"):
        """Returns noisy version of sample (after applying a gaussian noise
        `timesteps` times), as well as the noise itself."""
        # sample: [batch_size, C, H, W], values in [-1, 1]
        # timesteps: [batch_size, 1]

        noise = noise if noise is not None else torch.rand_like(sample)
        # noise = [batch_size, C, H, W]

        # added None(s) to turn the shape from [batch_size, 1] to
        # [batch_size, 1, 1, 1] so it can be broadcasted to sample shape,
        # [batch_size, C, H, W]. Alternatively, we could call unsqueeze(-1)
        # twice.
        #
        # a_coeff ^2 + b_coeff ^2 = 1
        a_coeff = self.sqrt_alphas_cumprod[timesteps, None, None]
        b_coeff = self.sqrt_one_minus_alphas_cumprod[timesteps, None, None]
        # a_coeff, b_coeff: [batch_size, 1, 1, 1]

        noisified = a_coeff.to(device) * sample.to(device) + b_coeff.to(
            device
        ) * noise.to(device)
        # noisified: [batch_size, C, H, W]

        return noisified, noise.to(device)
