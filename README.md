# ❄ Snowflakeizer ❄ 
A web application for generating novel, synthetic snowflake images using an unconditional DDPM U-Net model. It can generate images from a random latent space or in image-to-image mode, producing snowflakes based on an encoded and noised input image. 

## Features 
- Generate unique snowflake images from random latents.
- Image2Image mode: modify existing images to produce snowflake variations.
- Web interface (API) built with Flask for easy interaction.
- Set a constant seed and modify parameters for controlling the output image.

## Technologies
- Python 3.10
- Flask
- PyTorch
- 🤗 Diffusers
- HTML / CSS / JavaScript for the frontend
