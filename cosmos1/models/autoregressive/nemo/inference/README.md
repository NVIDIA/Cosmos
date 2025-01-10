# Cosmos Autoregressive-based World Foundation Models: NeMo Framework User Guide

Learn how to [run inference](#run-inference) with Cosmos Autoregressive-based World Foundation Models (WFMs) using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) for your custom Physical AI tasks by following this guide.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Autoregressive (AR) models. Review the available models and their compute requirements for post-training and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Inference | Multi-GPU Support |
|----------------------------------------------|------------------|------------------------------------------|---------|
| Cosmos-1.0-Autoregressive-4B                     | **Supported**    | 1 NVIDIA GPU*                            |    **Coming Soon**   |
| Cosmos-1.0-Autoregressive-12B                 | **Supported**    | 1 NVIDIA GPU*                            |   **Coming Soon**    |
| Cosmos-1.0-Autoregressive-5B-Video2World          | **Coming Soon**  |                                          |         |
| Cosmos-1.0-Autoregressive-13B-Video2World        | **Coming Soon**  |                                          |         |

**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

## Post-Training Inference Support Matrix

Cosmos Autoregressive-based WFMs can be post-trained for a variety of Physical AI tasks. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task  | Inference Support Status |
|-------------------------|--------------------|
| General post-training     | **Supported**      |
| Instruction control     | **Coming Soon**    |
| Action control          | **Coming Soon**    |
| Camera control          | **Coming Soon**    |
| Multi-view generation   | **Coming Soon**    |
| Multi-view generation with vehicle trajectory control | **Coming Soon** |

## Prerequisites

### 1. Review General Requirements

- System Configuration
  - **NVIDIA GPU and driver**: Ensure you have access to the minimum compute required to run the model(s), as listed in the model support matrix.
  - **Containerization Platform**: We recommend using Docker with NVIDIA Container Runtime (alternatively, you may use NVIDIA enroot).
- Get your [Hugging Face User Access Token](https://huggingface.co/docs/hub/en/security-tokens), which is required to obtain the Cosmos models for training and inference.
- Get your [Weights and Biases API Key](https://docs.wandb.ai/support/find_api_key/) for logging and tracking.

### 2. Clone the Cosmos Repository

```bash
git clone git@github.com:NVIDIA/Cosmos.git
```

### 3. Start the Container

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos AR models.

Run the following command to download and start the container:
   ```bash
   docker run --ipc=host -it --gpus=all \
    -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
    nvcr.io/nvidia/nemo:cosmos.1.0 bash
   ```

### 4. Download Checkpoints

To help you get started, we've provided a [download script](../download_autoregressive_nemo.py) to get the Cosmos Autoregressive checkpoints from Hugging Face. These checkpoints are in the NeMo distributed checkpoint format required to run post-training and inference with NeMo Framework.

1. Set the following environment variables:
   ```bash
   # You must set HF_HOME before running this script.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"
   ```
2. Run the following command to download the models:
   ```bash
   cd /workspace/Cosmos
   python cosmos1/models/autoregressive/nemo/download_autoregressive_nemo.py
   ```

## Run Inference

Running inference with Cosmos AR models lets you predict video frames and generate a new video that continues the scene from a given input video.

In this guide, we'll use this [example inference script](./general.py) to tokenize the input video into a sequence of tokens, which serve as prompts for the model. The model then generates new tokens representing the next set of frames. Finally, the new tokens are decoded back into video format. Only the last 9 frames of the input video are used to generate the next 24 frames.

### Run the Inference Script with Base Model

Complete the following steps to run inference on the 4B model.

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Path to the the mp4 file (In git-lfs)
   export INPUT_DATA=cosmos1/models/autoregressive/assets/v1p0/input.mp4
   ```
2. Run the following command:
   ```bash
   cd /workspace/Cosmos
   git lfs pull $INPUT_DATA

   NVTE_FLASH_ATTN=1 \
   NVTE_FUSED_ATTN=0 \
   NVTE_UNFUSED_ATTN=0 \
   torchrun --nproc-per-node 1 cosmos1/models/autoregressive/nemo/inference/general.py \
   --input_image_or_video_path $INPUT_DATA \
   --video_save_name "Cosmos-1.0-Autoregressive-4B.mp4"  \
   --ar_model_dir nvidia/Cosmos-1.0-Autoregressive-4B
   ```

### Run the Inference Script with Post-trained Model

Complete the following steps to generate a new output video.

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Inference with post-trained model.
   export NEMO_CHECKPOINT=./logs/default/checkpoints/epoch\=0-step\=19

   # Path to the the mp4 file (In git-lfs)
   export INPUT_DATA=cosmos1/models/autoregressive/assets/v1p0/input.mp4

   ```
2. Run the following command:
   ```bash
   cd /workspace/Cosmos
   git lfs pull $INPUT_DATA

   # change --ar_model_dir to a post-trained checkpoint under ./logs/default/checkpoints/
   NVTE_FLASH_ATTN=1 \
   NVTE_FUSED_ATTN=0 \
   NVTE_UNFUSED_ATTN=0 \
   torchrun --nproc-per-node 1 cosmos1/models/autoregressive/nemo/inference/general.py \
   --input_image_or_video_path $INPUT_DATA \
   --video_save_name "Cosmos-1.0-Autoregressive-4B.mp4" \
   --ar_model_dir "$NEMO_CHECKPOINT"
   ```

#### Example Output

The following output is an example video generated from the post-trained model using [`general.py`](./general.py):

<video src="https://github.com/user-attachments/assets/e744a5a4-2ce0-4de3-9497-7152b25c9022">
  Your browser does not support the video tag.
</video>

Generated videos are saved at the location configured in the `--video_save_name` parameter.

The input video used to generate this video can be found in `cosmos1/models/autoregressive/assets/v1p0/input.mp4`.

> **Disclaimer**:
> The post-training example in this documentation is a demonstration of general post-training and not a guaranteed recipe for success. Post-training outcomes depend heavily on the quality and diversity of the dataset. To achieve good results, ensure your dataset is clean, well-structured, diverse, and properly labeled. Poorly prepared data can lead to issues like overfitting, bias, or poor performance. Carefully curate your dataset to reflect the desired use case for reliable results.

### Configuration Options

The following table details the parameters that can be modified for accelerated inference with NeMo. You can adjust these parameters to optimize performance based on your specific requirements

| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--input_type`                | The input type (image or video)                                                  | `video` |
| `--input_image_or_video_path` | Path to the input video to run inference                                         | `cosmos1/models/autoregressive/assets/v1p0/input.mp4` |
| `--video_save_name`           | Path to generated video                                                          | `./nemo_generated_video.mp4` |
| `--ar_model_dir`              | Model name or path to model  `nvidia/Cosmos-1.0-Autoregressive-4B` or `nvidia/Cosmos-1.0-Autoregressive-12B`                                                    | `nvidia/Cosmos-1.0-Autoregressive-4B` |
| `--encoder_path`           | Path to encoder                                                        | `nvidia/Cosmos-1.0-Tokenizer-DV8x16x16` |
| `--decoder_path`           | Path to the decoder                                                         | `nvidia/Cosmos-1.0-Tokenizer-DV8x16x1"` |
| `--guardrail_dir`           | Path to guardrails                                                       | `nvidia/Cosmos-1.0-Guardrail` |
| `--top_p`                     | Top-p inference parameter                                                        | `0.9` |
| `--temperature`               | Sampling temperature                                                             | `1` |
| `--disable_diffusion_decoder` | Disables running diffusion decoder on the generated result                       | `False` |
