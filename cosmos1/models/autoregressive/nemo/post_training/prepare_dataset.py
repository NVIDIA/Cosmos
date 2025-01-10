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
from argparse import ArgumentParser
from glob import glob

import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

from cosmos1.models.autoregressive.nemo.utils import read_input_videos
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.utils import log

TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
DATA_RESOLUTION_SUPPORTED = [640, 1024]
NUM_CONTEXT_FRAMES = 33


def main(args):
    if args.encoder_path == "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16":
        args.encoder_path = os.path.join(snapshot_download(args.encoder_path), "encoder.jit")
    if args.decoder_path == "nvidia/Cosmos-1.0-Tokenizer-DV8x16x16":
        args.decoder_path = os.path.join(snapshot_download(args.decoder_path), "decoder.jit")
    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=NUM_CONTEXT_FRAMES,
    ).cuda()

    builders = {}
    key = "text"
    builders[key] = indexed_dataset.make_builder(
        f"{args.output_prefix}.bin",
        impl="mmap",
        chunk_size=64,
        pad_id=0,
        retrieval_db=None,
        vocab_size=64000,
        stride=64,
    )

    filepaths_final = glob(f"{args.input_videos_dir}/*.mp4")

    for filepath in filepaths_final:
        input_video = read_input_videos(filepath).cuda()
        batch_size, channels, frames, height, width = input_video.shape
        latent_shape = (
            (frames - 1) // TOKENIZER_COMPRESSION_FACTOR[0] + 1,
            height // TOKENIZER_COMPRESSION_FACTOR[1],
            width // TOKENIZER_COMPRESSION_FACTOR[2],
        )
        T, H, W = latent_shape
        video_tokenizer.latent_chunk_duration = T
        quantized_out, _ = video_tokenizer.encode(input_video, pixel_chunk_duration=None)
        indices = video_tokenizer.fsq_quantizer.codes_to_indices(quantized_out.permute(0, 2, 3, 4, 1))
        indices = rearrange(indices, "B T H W -> (B T H W)").detach().cpu()
        builders[key].add_item(torch.IntTensor(indices).detach().cpu())
        builders[key].end_document()

    builders[key].finalize(
        f"{args.output_prefix}.idx",
    )

    log.info(f"Stored the .bin and .idx files in {args.output_prefix}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_videos_dir", required=True, type=str, help="The path to the input videos")
    parser.add_argument(
        "--encoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to encoder"
    )
    parser.add_argument(
        "--decoder_path", default="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16", type=str, help="The path to the decoder"
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="The directory along with the output file name to write the .idx and .bin files (e.g /path/to/output/sample)",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
