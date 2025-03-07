# Cosmos Installation

We have only tested the installation with Ubuntu 24.04, 22.04, and 20.04.

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. Clone the repository.

```bash
git clone https://github.com/NVIDIA/Cosmos.git
cd Cosmos
```

3. Build a Docker image using `Dockerfile` and run the Docker container.

```bash
docker build -t cosmos .
docker run -d --name cosmos_container --gpus all --ipc=host -it -v $(pwd):/workspace cosmos
docker attach cosmos_container
```
