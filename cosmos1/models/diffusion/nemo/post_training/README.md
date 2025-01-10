# Cosmos Diffusion-based World Foundation Models: NeMo Framework User Guide

Learn how to [post-train](#post-train) Cosmos Diffusion-based World Foundation Models (WFMs) using the [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) for your custom Physical AI tasks by following this guide.

## Model Support Matrix

The NeMo Framework supports the following Cosmos Diffusion models. Review the available models and their compute requirements for post-tuning and inference to determine the best model for your use case.

| Model Name                               | Model Status | Compute Requirements for Post-Training |
|----------------------------------------------|------------------|------------------------------------------|
| Cosmos-1.0-Diffusion-7B-Text2World           | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Diffusion-14B-Text2World          | **Supported**    | 8 NVIDIA GPUs*                           |
| Cosmos-1.0-Diffusion-7B-Video2World          | **Coming Soon**  |                                          |
| Cosmos-1.0-Diffusion-14B-Video2WorldB        | **Coming Soon**  |                                          |


**\*** `H100-80GB` or `A100-80GB` GPUs are recommended.

## Post-Training Support Matrix

Cosmos Diffusion-based WFMs can be post-trained for a variety of Physical AI tasks. Review the following table for a list of available Physical AI post-training tasks:

| Post-training Task  | Post-Training Support Status |
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

## Post-train

Post-training a Cosmos Diffusion-based WFM enables you to train the model to generate videos that are more specific to your Physical AI use case.

For example, if you want to generate action sequences for a specific robot, you can post-train the model to generate videos that are more aligned with typical actions/outcomes for that robot.

There are 3 steps to post-training: preparing a dataset, preprocessing the data, and post-training the model.

### 1. Prepare a Dataset

The first step is to prepare a dataset. Post-training a Cosmos-1.0-Diffusion-Text2World-{7B/14B}-NeMo model enables you to generate videos of a specific subject in new environments using a collection of input videos of that same subject as reference material.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

Run the following command to download the sample videos used for post-training:

```bash
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir cosmos1/models/diffusion/assets/ --include "*.mp4*"
```

### 2. Preprocess Data

The second step is to preprocess the input videos. This generates the post-training samples and the metadata required for the post-training process by:

1. Selecting `N` chunks of 121 frames from each video, generating `N` post-training samples per video.
2. Encoding the 121 frames by first independently compressing the first frame and then applying an 8x temporal compression for the rest of the frames.
3. Generating `total_samples = # of videos x # of chunks` post-training samples.

Before proceeding, ensure all videos are in **RGB format**. Complete the following steps to generate the post-training samples and metadata for the robot dataset. Remember to follow the given prompt format by including the subject's name in the prompt. For example, if the subject is "robot," the prompt should read `"A video of sks robot."`.

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Path to Raw mp4 videos.
   export RAW_DATA="cosmos1/models/diffusion/assets/nemo_diffusion_example_data"

   # Path to Processed Dataset.
   export CACHED_DATA="./cached_data" && mkdir -p $CACHED_DATA
   ```
2. Run the following command to preprocess the data:
   ```bash
   python cosmos1/models/diffusion/nemo/post_training/prepare_dataset.py \
   --dataset_path $RAW_DATA \
   --output_path $CACHED_DATA \
   --prompt "A video of sks teal robot." \
   --num_chunks 500
   ```

Executing the [data preprocessing script](./prepare_dataset.py) generates the following files for each video (using `[i]` as the `index` of the video) at `$CACHED_DATA` path:

- **`[i].info.json`**: Metadata for the video sample.
- **`[i].t5_text_embeddings.pth`**: T5-generated text embedding for the video clip.
- **`[i].t5_text_mask.pth`**: Mask for T5 text embedding, set to all ones by default to use the entire text embedding.
- **`[i].video_latent.pth`**: 3D spatiotemporal video tokens generated from the video tokenizer.

### 3. Post-train the Model

The third step is to post-train the model. This step uses NeMo Framework's data and model parallelism capabilities to train the model on the post-training samples. This is accomplished by using utilizing Fully Sharded Data Parallel (FSDP) and Tensor Parallelism.

- **FSDP**: Distributes model parameters, optimizer states, and activations across all GPUs
- **Tensor Parallelism**: Spreads the parameter tensor of individual layers across GPUs.

> **NOTE**:
> For the 14B model, we also employ activation checkpointing to facilitate single-node training.

#### Run the Post-training Script

Complete the following steps to post-train the Cosmos-1.0-Diffusion-7B-Text2World model on the robot dataset using 8 GPUs.

1. Set the following environment variables:
   ```bash
   export HF_TOKEN="<your/HF/access/token>"
   export HF_HOME="<path/to/store/checkpoints>"

   # Optionally, you can monitor training progress with Weights and Biases (wandb).
   export WANDB_API_KEY="</your/wandb/api/key>"
   export WANDB_PROJECT_NAME="cosmos-diffusion-nemo-post-training"
   export WANDB_RUN_ID="cosmos_diffusion_7b_text2world_finetune"
   ```
2. Run the following command for Cosmos-Diffusion-Text2World-7B general post-training:
   ```bash
   NVTE_FUSED_ATTN=0 \
   CUDA_DEVICE_MAX_CONNECTIONS=1 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   torchrun --nproc_per_node=8 cosmos1/models/diffusion/nemo/post_training/general.py \
       --yes \
       --factory cosmos_diffusion_7b_text2world_finetune \
       data.path=$CACHED_DATA \
       trainer.max_steps=1000 \
       optim.config.lr=1e-6
   ```
3. You can now run inference with your post-trained model using the instructions [here](../inference/README.md#run-the-inference-script-with-post-trained-model).

#### Configuration Options

Before getting started, review the following parameters made available to the script. You can adjust these parameters to optimize performance based on your specific requirements.

| Parameter                      | Description                                                                     | Default |
|--------------------------------|---------------------------------------------------------------------------------|---------|
| `--factory`                   | recipe to use cosmos_diffusion_7b_text2world_finetune or cosmos_diffusion_14b_text2world_finetune for general post-training                                   | cosmos_diffusion_7b_text2world_finetune    |
| `data.path`                   | Path to processed post-training dataset (str).                                    | None    |
| `resume.restore_config.path`  | Path to pre-trained Cosmos Diffusion NeMo distributed checkpoint (str).         | None    |
| `optim.config.lr`             | Learning rate (float).                                                          | 1e-6    |
| `trainer.max_steps`           | Max number of post-training steps (int).                                             | 1000    |
| `log.log_dir`                 | Path to folder to save post-training logs and checkpoints (str).                     | None    |
