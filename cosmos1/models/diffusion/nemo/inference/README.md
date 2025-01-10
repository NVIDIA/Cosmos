# Cosmos Diffusion-based World Foundation Models: NeMo Framework User Guide

Learn how to [run inference](#inference) with Cosmos Diffusion-based World Foundation Models (WFMs) using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) for your custom Physical AI tasks by following this guide.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Diffusion models. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Inference | Multi-GPU Support
|----------------------------------------------|------------------|------------------------------------------|---------|
| Cosmos-1.0-Diffusion-7B-Text2World           | **Supported**    | 1 NVIDIA GPU*                            |    **Supported**   |
| Cosmos-1.0-Diffusion-14B-Text2World          | **Supported**    | 1 NVIDIA GPU*                            |    **Supported**   |
| Cosmos-1.0-Diffusion-7B-Video2World          | **Coming Soon**  |                                          |         |
| Cosmos-1.0-Diffusion-14B-Video2WorldB        | **Coming Soon**  |                                          |         |


**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

## Post-Trained Model Inference Support Matrix

Cosmos Diffusion-based WFMs can also be post-trained for a variety of Physical AI tasks and used for inference. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task      | Inference Support Status |
|-------------------------|--------------------|
| General post-training   | **Supported**      |
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

The [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) supports post-training and inference for Cosmos Diffusion models.

Run the following command to download and start the container:
```bash
docker run --ipc=host -it --gpus=all \
  -v $PATH_TO_COSMOS_REPO:/workspace/Cosmos \
  nvcr.io/nvidia/nemo:cosmos.1.0 bash
```

### 4. Download Checkpoints

To help you get started, we've provided a [download script](../download_diffusion_nemo.py) to get the Cosmos Diffusion checkpoints from Hugging Face. These checkpoints are in the NeMo distributed checkpoint format required to run post-training and inference with NeMo Framework.

1. Set the following environment variables:
   ```bash
   # You must set HF_HOME before running this script.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"
   ```
2. Run the following command to download the models:
   ```bash
   cd /workspace/Cosmos
   python cosmos1/models/diffusion/nemo/download_diffusion_nemo.py
   ```

## Run Inference

Running inference with Cosmos Diffusion models lets you generate a video conditioned on a text prompt.

Our inference script enables accelerated world generation with context parallel. We use context parallelism to split the diffusion process across multiple GPUs, providing near-linear scaling efficiency. Our diffusion pipeline also allows the user to set a variety of hyperparameters including the random seed, classifier-free guidance scale, negative prompt, video resolution, and video fps.

General post-training is essentially a continuation of pre-training. To perform inference with models that have been post-trained with general post-training, you can set the `subject_name` parameter to the subject the model was post-trained on. The `prompt` parameter is then used to describe the setting and events in the generated world. The final prompt will be "A video of sks `{subject_name}`. `{prompt}`". We can also use [inference/general.py](./general.py) to perform inference on the base model since the model architecture is the same as the general post-trained models.

We also provide the option to upsample the `prompt` and make it more detailed. This can improve the quality of the generated world.

### Run the Inference Script with Base Model

Complete the following steps to generate a new output video of a robot cooking.

1. Set the following environment variables:
   ```bash
   # HuggingFace Cache to save T5 text encoder, video tokenizer, prompt upsampler, and guardrails weights.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Number of GPU devices available for inference. Supports up to 8 GPUs for accelerated inference.
   export NUM_DEVICES=1

   # Prompt describing world scene and actions taken by subject (if provided).
   export PROMPT="The teal robot is cooking food in a kitchen. Steam rises from a simmering pot as the robot chops vegetables on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered measuring spoons and a half-empty bottle of olive oil."
   ```
2. Run the following command:
   ```bash
   NVTE_FUSED_ATTN=0 \
   torchrun --nproc_per_node=$NUM_DEVICES cosmos1/models/diffusion/nemo/inference/general.py \
       --model Cosmos-1.0-Diffusion-7B-Text2World \
       --cp_size $NUM_DEVICES \
       --num_devices $NUM_DEVICES \
       --video_save_path "Cosmos-1.0-Diffusion-7B-Text2World.mp4" \
       --guidance 7 \
       --seed 1 \
       --prompt "$PROMPT" \
       --enable_prompt_upsampler
   ```

### Run the Inference Script with Post-trained Model

Complete the following steps to generate a new output video from the model post-trained with general post-training.

1. Set the following environment variables:
   ```bash
   # HuggingFace Cache to save T5 text encoder, video tokenizer, prompt upsampler, and guardrails weights.
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Inference with post-trained model. Find post-trained model under nemo_experiments. Example path:
   export NEMO_CHECKPOINT=nemo_experiments/cosmos_diffusion_7b_text2world_finetune/default/2024-12-17_01-28-03/checkpoints/epoch=39-step=199/weights

   # Number of GPU devices available for inference. Supports up to 8 GPUs for accelerated inference.
   export NUM_DEVICES=1

   # Prompt describing world scene and actions taken by subject (if provided).
   export PROMPT="The teal robot is cooking food in a kitchen. Steam rises from a simmering pot as the robot chops vegetables on a worn wooden cutting board. Copper pans hang from an overhead rack, catching glints of afternoon light, while a well-loved cast iron skillet sits on the stovetop next to scattered measuring spoons and a half-empty bottle of olive oil."
   ```
2. Run the following command:
   ```bash
   NVTE_FUSED_ATTN=0 \
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/inference/general.py \
       --model Cosmos-1.0-Diffusion-7B-Text2World \
       --nemo_checkpoint "$NEMO_CHECKPOINT" \
       --cp_size $NUM_DEVICES \
       --num_devices $NUM_DEVICES \
       --video_save_path "Cosmos-1.0-Diffusion-7B-Text2World.mp4" \
       --guidance 7 \
       --seed 1 \
       --prompt "$PROMPT" \
       --subject_name "teal robot"  \
       --enable_prompt_upsampler
   ```

#### Example Output

The following output is an example video generated from the post-trained model using [`general.py`](./inference/general.py):

<video src="https://github.com/user-attachments/assets/a2b5bc7e-4e0a-4514-a04e-919281cee6fa">
  Your browser does not support the video tag.
</video>

Generated videos are saved at the location configured in the `SAVE_PATH` parameter.

> **Tip**:
> For faster inference, you can remove the `--enable_prompt_upsampler` parameter, but this may degrade the generated result.

> **Disclaimer**:
> The post-training example in this documentation is a demonstration of general post-training and not a guaranteed recipe for success. Post-training outcomes depend heavily on the quality and diversity of the dataset. To achieve good results, ensure your dataset is clean, well-structured, diverse, and properly labeled. Poorly prepared data can lead to issues like overfitting, bias, or poor performance. Carefully curate your dataset to reflect the desired use case for reliable results.

### Configuration Options

The following table details the parameters that can be modified for accelerated inference with NeMo. You can adjust these parameters to optimize performance based on your specific requirements. The model inference hyperparameters listed below have the same functionality as in [Cosmos Diffusion Common Parameters](cosmos1/models/diffusion/README.md#parameters).


| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--model`                     | Name of Cosmos Text2World Diffusion model to use for inference.                  | `Cosmos-1.0-Diffusion-7B-Text2World` |
| `--prompt`                    | Prompt which the sampled video is conditioned on. Tries to generate what is mentioned in the prompt. | *None* (user must provide) |
| `--negative_prompt`           | Negative prompt for improved quality                                             | "The video captures a series of frames showing ugly scenes..." |
| `--subject_name`              | Name of the subject the model was post-trained on. This can be left empty for base model inference. | *None* |
| `--guidance`                  | A control mechanism that determines how closely the model follows specified conditions (prompt) during the generation process. We recommend starting with a guidance of 7 and increasing it later if necessary. | 7 |
| `--sampler`                   | Sampling method used for generation. Only supports **RES** sampler from [this paper](https://arxiv.org/pdf/2308.02157). | `RES` |
| `--video_save_path`           | Location to save generated videos.                                               | `Cosmos-1.0-Diffusion-7B-Text2World.mp4` |
| `--fps`                       | Frames-per-second of generated video. Cosmos Diffusion models generate videos at 24 FPS by default. | 24 |
| `--height`                    | Height of the generated video. Set to 704 pixels by default, which is the largest supported video height for Cosmos Diffusion. | 704 |
| `--width`                     | Width of the generated video. Set to 1280 pixels by default, which is the largest supported video width for Cosmos Diffusion. | 1280 |
| `--seed`                      | Random seed for generating initial noise sample. Changing this will create a different video for the same prompt. Keep the seed fixed to maintain deterministic video generations. | 1 |
| `--num_devices`               | [1–8] Number of GPUs to use in parallel for inference.                           | 8 |
| `--cp_size`                   | [1–8] Number of context parallel ranks to spawn for parallelized inference. Must be equal to `num_devices`. | 8 |
