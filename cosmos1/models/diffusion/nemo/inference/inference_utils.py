# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import imageio
import numpy as np
import torch

from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.diffusion.prompt_upsampler.text2world_prompt_upsampler_inference import (
    create_prompt_upsampler,
    run_chat_completion,
)
from cosmos1.models.guardrail.common.presets import (
    create_text_guardrail_runner,
    create_video_guardrail_runner,
    run_text_guardrail,
    run_video_guardrail,
)
from cosmos1.utils import log


def get_upsampled_prompt(
    prompt_upsampler_model: AutoRegressiveModel, input_prompt: str, temperature: float = 0.01
) -> str:
    """
    Get upsampled prompt from the prompt upsampler model instance.

    Args:
        prompt_upsampler_model: The prompt upsampler model instance.
        input_prompt (str): Original prompt to upsample.
        temperature (float): Temperature for generation (default: 0.01).

    Returns:
        str: The upsampled prompt.
    """
    dialogs = [
        [
            {
                "role": "user",
                "content": f"Upsample the short caption to a long caption: {input_prompt}",
            }
        ]
    ]

    upsampled_prompt = run_chat_completion(prompt_upsampler_model, dialogs, temperature=temperature)
    return upsampled_prompt


def print_rank_0(string: str):
    rank = torch.distributed.get_rank()
    if rank == 0:
        log.info(string)


def process_prompt(
    prompt: str,
    checkpoint_dir: str,
    prompt_upsampler_dir: str,
    guardrails_dir: str,
    image_path: str = None,
    enable_prompt_upsampler: bool = True,
) -> str:
    """
    Handle prompt upsampling if enabled, then run guardrails to ensure safety.

    Args:
        prompt (str): The original text prompt.
        checkpoint_dir (str): Base checkpoint directory.
        prompt_upsampler_dir (str): Directory containing prompt upsampler weights.
        guardrails_dir (str): Directory containing guardrails weights.
        image_path (str, optional): Path to an image, if any (not implemented for upsampling).
        enable_prompt_upsampler (bool): Whether to enable prompt upsampling.

    Returns:
        str: The upsampled prompt or original prompt if upsampling is disabled or fails.
    """

    text_guardrail = create_text_guardrail_runner(os.path.join(checkpoint_dir, guardrails_dir))

    # Check if the prompt is safe
    is_safe = run_text_guardrail(str(prompt), text_guardrail)
    if not is_safe:
        raise ValueError("Guardrail blocked world generation.")

    if enable_prompt_upsampler:
        if image_path:
            raise NotImplementedError("Prompt upsampling is not supported for image generation")
        else:
            prompt_upsampler = create_prompt_upsampler(
                checkpoint_dir=os.path.join(checkpoint_dir, prompt_upsampler_dir)
            )
            upsampled_prompt = get_upsampled_prompt(prompt_upsampler, prompt)
            print_rank_0(f"Original prompt: {prompt}\nUpsampled prompt: {upsampled_prompt}\n")
            del prompt_upsampler

            # Re-check the upsampled prompt
            is_safe = run_text_guardrail(str(upsampled_prompt), text_guardrail)
            if not is_safe:
                raise ValueError("Guardrail blocked world generation.")

            return upsampled_prompt
    else:
        return prompt


def save_video(
    grid: np.ndarray,
    fps: int,
    H: int,
    W: int,
    video_save_quality: int,
    video_save_path: str,
    checkpoint_dir: str,
    guardrails_dir: str,
):
    """
    Save video frames to file, applying a safety check before writing.

    Args:
        grid (np.ndarray): Video frames array [T, H, W, C].
        fps (int): Frames per second.
        H (int): Frame height.
        W (int): Frame width.
        video_save_quality (int): Video encoding quality (0-10).
        video_save_path (str): Output video file path.
        checkpoint_dir (str): Directory containing model checkpoints.
        guardrails_dir (str): Directory containing guardrails weights.
    """
    video_classifier_guardrail = create_video_guardrail_runner(os.path.join(checkpoint_dir, guardrails_dir))

    # Safety check on the entire video
    grid = run_video_guardrail(grid, video_classifier_guardrail)

    kwargs = {
        "fps": fps,
        "quality": video_save_quality,
        "macro_block_size": 1,
        "ffmpeg_params": ["-s", f"{W}x{H}"],
        "output_params": ["-f", "mp4"],
    }

    imageio.mimsave(video_save_path, grid, "mp4", **kwargs)
