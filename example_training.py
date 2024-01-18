import os
import torch
from torchvision.utils import save_image

from diffusion_pytorch import dataset, gaussian_diffusion, trainer, unet

TRAINING_DATA_PATH = "./diffusion_pytorch/trainingdata"
CHECKPOINT_PATH = "./checkpoint.pt"


if __name__ == "__main__":
    d = dataset.Dataset(imgs_dir=TRAINING_DATA_PATH)
    diffusion = gaussian_diffusion.GaussianDiffusion(
        model=unet.UNet(), max_timesteps=100
    )

    tr = trainer.Trainer(
        diffusion=diffusion,
        dataset=d,
        train_batch_size=32,
        train_lr=8e-5,
        train_epochs=4,
        train_num_steps=1_000,
        save_every_n_steps=10,
        save_folder=CHECKPOINT_PATH,
        use_gpu=True,
    )

    # Start from last checkpoint (if any).
    if os.path.exists(CHECKPOINT_PATH):
        tr.load_checkpoint(CHECKPOINT_PATH)

    tr.train()
