# diffusion-pytorch
Denoising Diffusion Probabilistic Models in pytorch

todos:
  - proper support for gpu/device
 

## Usage 

```python
import torch

from gaussian_diffusion import GaussianDiffusion
from unet import UNet

diffusion = GaussinDiffusion(
    model=UNet(),
    max_timesteps=500
)

batch_size, channel, height, width = 10, 3, 128, 128
imgs = torch.rand([batch_size, channel, height, width]) # normalized to [0, 1]
loss = diffusion(imgs)
loss.backward()

# After many training loops

generated_img = diffusion.sample(batch_size=3)

```

or use `Trainer` module

```python
import os
import torch
from torchvision.utils import save_image

from diffusion_pytorch import dataset, gaussian_diffusion, trainer, unet

TRAINING_DATA_PATH = "path/to/training/images"
CHECKPOINT_PATH = "checkpoint/path/checkpoint.pt"


if __name__ == "__main__":
    d = dataset.Dataset(imgs_dir=TRAINING_DATA_PATH)
    diffusion = gaussian_diffusion.GaussianDiffusion(
        model=unet.UNet(), max_timesteps=500
    )

    tr = trainer.Trainer(
        diffusion=diffusion,
        dataset=d,
        train_batch_size=32,
        train_lr=8e-5,
        train_epochs=4,
        train_num_steps=100_000,
        save_every_n_steps=1000,
        save_folder=CHECKPOINT_PATH,
    )

    # Start from last checkpoint (if any).
    if os.path.exists(CHECKPOINT_PATH):
        tr.load_checkpoint(CHECKPOINT_PATH)

    tr.train()

    # Generates 3 images (in tensor format).
    generated_imgs = diffusion.sample(batch_size=3)
```

or

```bash
python3 -m venv myvenv
source myvenv/bin/activate

git clone https://github.com/m6n3/diffusion-pytorch.git
cd diffusion-pytorch
python -m pip install --upgrade pip
python -m pip install --upgrade wheel
python -m pip install --upgrade .
pip install -r requirements.txt

# Optional: run all the tests
nose2 -v

# Example training script (saves checkpoint at ./checkpoint.pt).
python3 example_training.py

# Example gen script (needs checkpoint at ./checkpoint.pt).
python3 example_gen.py   # saves "img.png" into current directory.
```


