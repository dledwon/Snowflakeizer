import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL

from PIL import Image

from tqdm import tqdm
import copy
from pathlib import Path
import json


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


@torch.no_grad()
def update_ema(model, ema_model, decay=0.999):
    model_params = dict(model.named_parameters())
    ema_params = dict(ema_model.named_parameters())

    for name, ema_p in ema_params.items():
        p = model_params[name].detach().float().cpu()
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


class SnowflakeDataset(Dataset):
    def __init__(self, folder, size=512):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.9, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


CONFIG_PATH = Path("config/config.json")

DEVICE = "cuda"
EMA_DECAY = 0.995
NUM_EPOCHS = 200
LEARNING_RATE = 2e-5

with open(CONFIG_PATH) as f:
    config = json.load(f)
DATA_DIR = Path(config["train_data_dir"])
MODEL_DIR = Path(config["train_checkpoint_dir"])

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(DEVICE)
vae.eval()
vae.requires_grad_(False)

unet = UNet2DModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(DEVICE)

ema_unet = copy.deepcopy(unet).cpu()
ema_unet.eval()
for par in ema_unet.parameters():
    par.requires_grad_(False)

unet.train()

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2"
)

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=LEARNING_RATE
)

dataset = SnowflakeDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

scaling = 0.18215

best_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for images in tqdm(loader):
        images = images.to(DEVICE)

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * scaling

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=DEVICE
        ).long()

        noisy_latents = noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        noise_pred = unet(
            noisy_latents,
            timesteps
        ).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        if epoch > 5:
            update_ema(unet, ema_unet, decay=EMA_DECAY)

        epoch_loss += loss.item()

    epoch_loss /= len(loader)

    print(f"Epoch {epoch} | avg loss {epoch_loss:.4f}")

    if epoch > 10 and epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            {
                # "unet": unet.state_dict(),
                "ema": {n: p.clone() for n, p in ema_unet.named_parameters()}
            },
            Path.joinpath(MODEL_DIR, "ema_best.pt")
        )
        print(f"--> Current best model saved: {best_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                # "unet": unet.state_dict(),
                "ema": {n: p.clone() for n, p in ema_unet.named_parameters()}
            },
            Path.joinpath(MODEL_DIR, f"ema_epoch_{epoch + 1}.pt")
        )
