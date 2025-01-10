# Cosmos Post-training

In the [Cosmos paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai), we discuss several post-training examples of Cosmos pre-trained World Foundation Models (WFMs) for various Physical AI tasks, including

- General Post-Training: Fine-tune the WFM to generate a target distribution of videos based on the custom dataset. The target distribution could include a specific camera spec or a specific domain such as a factory.
- Instruction Control: Post-trains models for robotic manipulation to predict videos based on textual instructions, enabling robots to visually simulate tasks like folding clothes or picking up objects.
- Action Control: Post-trains models for robotic manipulation to predict the next visual frame based on action vectors, simulating robotic tasks like object handling or movement planning.
- Camera Control: Adds camera pose conditioning to generate 3D-consistent video simulations from single images, enabling joystick-like navigation in virtual environments.
- Multi-View Generation: Post-trains models for autonomous vehicles to generate synchronized multi-view videos from text prompts, simulating driving scenarios with multiple camera perspectives.
- Multi-View Generation with Vehicle Trajectory Control: Extends multi-view generation by incorporating trajectory inputs, enabling precise simulation of driving environments for autonomous vehicles, adhering to specified paths.

Except for the instruction control where the WFM is post-trained on a dataset of instruction-video pairs, all other cases require minor modifications of the network architectures. Post-training tasks will be supported by NeMo Framework. In this initial release, we provide post-training scripts for the general post-training of both diffusion and autorgressive WFMs. Scripts of the other post-training tasks will be provided in a future release.

## Post-training Support Matrix

| Post-training Task  | Diffusion WFM | Autoregressive WFM |
|---------------------|---------------|--------------------|
| General post-training | [Supported](../models/diffusion/nemo/post_training/README.md) | [Supported](../models/autoregressive/nemo/post_training/README.md) |
| Instruction control | Coming soon | Coming soon |
| Action control | Coming soon | Coming soon |
| Camera control | Coming soon | Coming soon |
| Multi-view generation | Coming soon | Coming soon |
| Multi-view generation with vehicle trajectory control | Coming soon | Coming soon |
