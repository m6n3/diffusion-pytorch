import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        *,
        diffusion,
        dataset,
        train_batch_size=32,
        train_lr=8e-5,
        train_epochs=10,
        train_num_steps=1_000_000,
        save_every_n_steps=5,
        save_folder="./model",
        use_gpu=False,
    ):
        super().__init__()

        self.diffusion = diffusion
        self.dataloader = DataLoader(
            dataset=dataset, batch_size=train_batch_size, shuffle=True, drop_last=True
        )
        self.train_epochs = train_epochs
        self.train_num_steps = train_num_steps
        self.optim = Adam(diffusion.get_model().parameters(), lr=train_lr)
        self.save_every_n_steps = save_every_n_steps
        self.save_folder = save_folder

        if use_gpu:
            assert (
                torch.cuda.is_available()
            ), "Error: no GPU device is available, consider setting `use_gpu` to False."
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.diffusion.set_device_to(self.device)

    def load_checkpoint(self, checkpoint_path):
        self.diffusion.load_checkpoint(checkpoint_path)

    def train(self):
        self.diffusion.get_model().train()
        best_loss = float("inf")
        num_steps = 0
        for epoch in range(self.train_epochs):
            running_loss = 0.0
            steps_in_running_loss = 0
            for idx, imgs in enumerate(tqdm(self.dataloader)):
                num_steps += 1
                if num_steps > self.train_num_steps:
                    break

                self.optim.zero_grad()
                loss = self.diffusion(imgs)
                loss.backward()
                self.optim.step()

                running_loss += loss
                steps_in_running_loss += 1

                if (
                    self.save_folder
                    and (
                        idx % self.save_every_n_steps == 0
                        or num_steps == self.save_every_n_steps
                    )
                    and (running_loss / steps_in_running_loss) < best_loss
                ):
                    best_loss = running_loss / steps_in_running_loss
                    torch.save(
                        self.diffusion.get_model().state_dict(), self.save_folder
                    )
                    running_loss = 0.0
                    steps_in_running_loss = 0
