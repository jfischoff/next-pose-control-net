#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset, load_from_disk
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed

import glob

from diffusers import (
    FlaxAutoencoderKL,
    FlaxControlNetModel,
    FlaxDDPMScheduler,
    FlaxStableDiffusionControlNetPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from mod_pipeline import FlaxStableDiffusionMultiControlNetPipeline

# To prevent an error that occurs when there are abnormally large compressed data chunk in the png image
# see more https://github.com/python-pillow/Pillow/issues/5610
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = logging.getLogger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(base_controlnet, base_controlnet_params, controlnet, controlnet_params, tokenizer, args, rng, weight_dtype, fs=1):
    logger.info("Running validation... ")

    pipeline, params = FlaxStableDiffusionMultiControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        controlnet=[base_controlnet, controlnet],
        safety_checker=None,
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    params = jax_utils.replicate(params)

    params["controlnet"] = [jax_utils.replicate(base_controlnet_params), controlnet_params]
    
    num_samples = jax.device_count()
    prng_seed = jax.random.split(rng, jax.device_count())

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
        pose_validation_images = args.pose_validation_image
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )


    
    for vid_idx, (validation_prompt, validation_image, pose_validation_image_dir) in enumerate(zip(validation_prompts, validation_images, pose_validation_images)):

        validation_image = Image.open(validation_image).convert("RGB")
        processed_image = num_samples * [validation_image]
        processed_image = torch.stack([conditioning_image_transforms(image) for image in processed_image], dim=0)

        
        prompts = num_samples * [validation_prompt]
        prompt_ids = pipeline.prepare_text_inputs(prompts)
        prompt_ids = shard(prompt_ids)

        
        for frame_idx, pose_validation_image in enumerate(sorted(glob.glob(pose_validation_image_dir+'/*.png'))[::fs]):

            
            pose_validation_image = Image.open(pose_validation_image).convert("RGB")
            pose_processed_image = num_samples * [pose_validation_image]
            pose_processed_image = torch.stack([conditioning_image_transforms(image) for image in pose_processed_image], dim=0)
  
            sharded_processed_pose_image = shard(pose_processed_image.float().numpy())

            all_processed_image = torch.cat([processed_image, pose_processed_image], dim=1).float().numpy()

            sharded_all_processed_image = shard(all_processed_image)

            frame_images = pipeline(
                prompt_ids=prompt_ids,
                image=[sharded_processed_pose_image, sharded_all_processed_image],
                params=params,
                prng_seed=prng_seed,
                num_inference_steps=50,
                jit=True,
                controlnet_conditioning_scale = [0.8, 0.2],
            ).images

            frame_images = frame_images.reshape((frame_images.shape[0] * frame_images.shape[1],) + frame_images.shape[-3:])

            images = pipeline.numpy_to_pil(frame_images)

            for sample_idx, image in enumerate(images):
                image.save(f'{args.output_dir}/out_{vid_idx}_{sample_idx}_{frame_idx}.png')

                
        
    return image_logs



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--base_controlnet_model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--base_revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--base_controlnet_from_pt",
        action="store_true",
        help="Load the pretrained model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--base_controlnet_revision",
        type=str,
        default=None,
        help="Revision of controlnet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--from_pt",
        action="store_true",
        help="Load the pretrained model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--controlnet_revision",
        type=str,
        default=None,
        help="Revision of controlnet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="How many training steps to profile in the beginning.",
    )
    parser.add_argument(
        "--profile_validation",
        action="store_true",
        help="Whether to profile the (last) validation.",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Whether to dump an initial (before training loop) and a final (at program end) memory profile.",
    )
    parser.add_argument(
        "--ccache",
        type=str,
        default=None,
        help="Enables compilation cache.",
    )
    parser.add_argument(
        "--controlnet_from_pt",
        action="store_true",
        help="Load the controlnet model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/{timestamp}",
        help="The output directory where the model predictions and checkpoints will be written. "
        "Can contain placeholders: {timestamp}.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--pose_validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet pose conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide a matching number of `--validation_prompt`s, a"
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` and logging the images."
        ),
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=100,
        help=(
            "Generate image every X input frames."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    args.output_dir = args.output_dir.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S"))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args





def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()


    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(0)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    else:
        raise NotImplementedError("No tokenizer specified!")


    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
        from_pt=args.from_pt,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )

    logger.info("Loading existing base controlnet weights")
    base_controlnet, base_params = FlaxControlNetModel.from_pretrained(
        args.base_controlnet_model_name_or_path,
        revision=args.base_controlnet_revision,
        from_pt=args.base_controlnet_from_pt,
        dtype=weight_dtype,
    )

    
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            revision=args.controlnet_revision,
            from_pt=args.controlnet_from_pt,
            dtype=jnp.float32,
        )
    else:
        logger.info("No controlnet weights specified!")
        rng, rng_params = jax.random.split(rng)

        controlnet = FlaxControlNetModel(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            attention_head_dim=unet.config.attention_head_dim,
            cross_attention_dim=unet.config.cross_attention_dim,
            use_linear_projection=unet.config.use_linear_projection,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
        )
        controlnet_params = controlnet.init_weights(rng=rng_params)
        controlnet_params = unfreeze(controlnet_params)
        for key in [
            "conv_in",
            "time_embedding",
            "down_blocks_0",
            "down_blocks_1",
            "down_blocks_2",
            "down_blocks_3",
            "mid_block",
        ]:
            controlnet_params[key] = unet_params[key]
        rng, rng_params = jax.random.split(rng)

        controlnet_params['controlnet_cond_embedding']['conv_in']['kernel']=jax.nn.initializers.lecun_normal()(rng_params, (3,3,6,16), jnp.float32)
            


    # Initialize our training
    validation_rng, train_rngs = jax.random.split(rng)

    # Replicate the train state on each device
    unet_params = jax_utils.replicate(unet_params)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)
    controlnet_params = jax_utils.replicate(controlnet_params)

    image_logs = log_validation(base_controlnet, base_params, controlnet, controlnet_params, tokenizer, args, validation_rng, weight_dtype)



if __name__ == "__main__":
    main()

