import os

import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler
from torchvision import transforms

from PIL import Image

import io
import json
import base64
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


CONFIG_PATH = Path("config/config.json")

DEVICE = "cuda"
DTYPE = torch.float32
LATENT_SCALE = 0.18215

with open(CONFIG_PATH) as f:
    config = json.load(f)

print("Loading VAE and EMA UNet...")
# VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device=DEVICE, dtype=DTYPE)
vae.eval()
vae.requires_grad_(False)

# UNet EMA
MODEL_PATH = Path(config["model_path"])
ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

unet_ema = UNet2DModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device=DEVICE, dtype=DTYPE)

unet_ema.load_state_dict(
    {k: v.to(device=DEVICE, dtype=DTYPE) for k, v in ckpt["ema"].items()}
)
unet_ema.eval()

print("VAE + EMA UNet loaded")


def generate_random(steps, manual_seed=0):
    if manual_seed != 0:
        torch.manual_seed(manual_seed)

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False
    )
    scheduler.set_timesteps(steps)

    latent = torch.randn((1, 4, 64, 64), device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet_ema(latent, t).sample
            latent = scheduler.step(noise_pred, t, latent).prev_sample

        image = vae.decode(latent / LATENT_SCALE).sample
        image = (image.clamp(-1, 1) + 1) / 2  # 0-1 range
    return image


def generate_img2img(input_file, steps, strength, manual_seed=0):
    if manual_seed != 0:
        torch.manual_seed(manual_seed)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False
    )
    scheduler.set_timesteps(steps)

    img = Image.open(input_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        latent_dist = vae.encode(img_tensor).latent_dist
        latents = latent_dist.sample() * LATENT_SCALE

    t_enc = int(strength * scheduler.num_train_timesteps)
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(latents, noise, torch.tensor([t_enc], device=DEVICE))

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet_ema(latents, t).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        image = vae.decode(latents / LATENT_SCALE).sample
        image = (image.clamp(-1, 1) + 1) / 2

    return image


def images_to_base64(image):
    buffer = io.BytesIO()
    save_image(image, buffer, nrow=4, format="png")  # <- dodaj format
    buffer.seek(0)
    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"
