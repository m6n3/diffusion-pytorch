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
import torch

from dataset import Dataset
from gaussian_diffusion import GaussianDiffusion
from unet import UNet

dataset = Dataset(imgs_dir='/path/to/images')
diffusion = GaussinDiffusion(
    model=UNet(),
    max_timesteps=500
)

trainer = Trainer(
  diffusion=diffusion,
  dataset=dataset,
  train_batch_size=32,
  train_lr=8e-5,
  train_epoch=4,
  train_num_steps=100_000,
  save_every_n_steps=5,
  save_folder="./model"
)

trainer.Train()

generated_img = diffusion.sample(batch_size=3)

```


