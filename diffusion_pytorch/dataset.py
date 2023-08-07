import os
import torch

from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, transforms=None):
        self.imgs_dir = imgs_dir
        self.transforms = transforms
        self.imgs = [
            os.path.join(imgs_dir, f)
            for f in os.listdir(imgs_dir)
            if os.path.isfile(os.path.join(imgs_dir, f)) and self._is_img_file(f)
        ]

    def _is_img_file(self, path):
        exts = ["jpg", "jpeg", "png"]
        return any(path.endswith(ext) for ext in exts)

    def default_transforms(img_size=64):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                # To make model less prone to overfitting
                transforms.RandomHorizontalFlip(),
                # Convertsto a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transforms(img) if self.transforms else img
