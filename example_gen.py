import os
import torch
from torchvision.utils import save_image

from diffusion_pytorch import dataset, gaussian_diffusion, trainer, unet

TRAINING_DATA_PATH = "./diffusion_pytorch/trainingdata"
CHECKPOINT_PATH = "./checkpoint.pt"
OUTPUT_PATH = "./img.png"

if __name__ == "__main__":
    diffusion = gaussian_diffusion.GaussianDiffusion(
        model=unet.UNet(), max_timesteps=100
    )

    assert os.path.exists(CHECKPOINT_PATH)
    diffusion.load_checkpoint(CHECKPOINT_PATH)

    # Generate one image.
    img_tensor = diffusion.sample(batch_size=1)
    save_image(img_tensor, OUTPUT_PATH)
