import argparse
import os
import random

import cv2
import gradio as gr
import numpy as np
import torch
# from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from unhcv.common.utils import obj_load, obj_dump

from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


# torch.set_grad_enabled(False)

SD_PATH = "/home/zhuyixing/model/PowerPaint-v2-1/realisticVisionV60B1_v51VAE"
SD_INPAINTING_PATH = "/home/tiger/model/stable-diffusion-inpainting"
checkpoint_dir = "/home/zhuyixing/model/PowerPaint-v2-1"
weight_dtype = torch.float16
local_files_only = True
unet = UNet2DConditionModel.from_pretrained(
    SD_PATH,
    subfolder="unet",
    revision=None,
    torch_dtype=weight_dtype,
    local_files_only=local_files_only,
)
text_encoder_brushnet = CLIPTextModel.from_pretrained(
    SD_PATH,
    subfolder="text_encoder",
    revision=None,
    torch_dtype=weight_dtype,
    local_files_only=local_files_only,
)
brushnet = BrushNetModel.from_unet(unet)
base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    text_encoder_brushnet=text_encoder_brushnet,
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=False,
    safety_checker=None,
)
pipe.unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    revision=None,
    torch_dtype=weight_dtype,
    local_files_only=local_files_only,
)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained=base_model_path,
    subfolder="tokenizer",
    revision=None,
    torch_type=weight_dtype,
    local_files_only=local_files_only,
)

# add learned task tokens into the tokenizer
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder_brushnet,
    placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
    initialize_tokens=["a", "a", "a"],
    num_vectors_per_token=10,
)
load_model(
    pipe.brushnet,
    os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
)

pipe.text_encoder_brushnet.load_state_dict(
    torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")
