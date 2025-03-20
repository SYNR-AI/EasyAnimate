import os

import numpy as np
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       FlowMatchEulerDiscreteScheduler, PNDMScheduler)
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from diffusers.utils import export_to_video
from safetensors.torch import load_file

model_name = "/cv/models/Wan2.1-T2V-14B-Diffusers/"

transformer_path = 'output_dir/wan_movie_4k/checkpoint-100/transformer/diffusion_pytorch_model.safetensors'
transformer3d = WanTransformer3DModel.from_pretrained(model_name, subfolder="transformer", torch_dtype=torch.bfloat16)
state_dict = load_file(transformer_path)
transformer3d.load_state_dict(state_dict)

vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_name, transformer_3d=transformer3d, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=256,
    width=480,
    num_frames=49,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=15)
