import os

import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL, UNet2DModel, DDIMScheduler
from torchvision import transforms

from PIL import Image

import io
import base64

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

DEVICE = "cuda"
DTYPE = torch.float32
LATENT_SCALE = 0.18215

print("Loading VAE and EMA UNet...")
# VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device=DEVICE, dtype=DTYPE)
vae.eval()
vae.requires_grad_(False)

# UNet EMA
unet_ema = UNet2DModel.from_pretrained(
    "dledwon/latent-ddpm-unet-ema-snowflakes",
    use_safetensors=True
).to(device=DEVICE, dtype=DTYPE)
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
