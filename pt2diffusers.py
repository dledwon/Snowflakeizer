from diffusers import UNet2DModel
import torch

DEVICE = "cuda"

unet = UNet2DModel(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(DEVICE)

ckpt = torch.load(r"\\157.158.92.32\awmd4s\snowflakes\models\ema_best.pt")
unet.load_state_dict(ckpt["ema"])

unet.save_pretrained(r"\\157.158.92.32\awmd4s\snowflakes\models\latent-ddpm-ema-snowflakes")
