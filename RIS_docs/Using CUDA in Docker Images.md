
[Compute1](../Compute1.md)

# Using CUDA in Docker Images

- [Overview](#overview)
- [Using the Correct Version](#using-the-correct-version)
- [Testing Your Image](#testing-your-image)

> [!IMPORTANT]
> Compute Resources
>
> - Have questions or need help with compute, including activation or issues? Follow [this link.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43)
> - [RIS Services Policies](../RIS%20Services%20Policies.md)

> [!IMPORTANT]
> Docker Usage
>
> - The information on this page assumes that you have a knowledge base of using Docker to create images and push them to a repository for use. If you need to review that information, please see the links below.
> - [Docker and the RIS Compute1 Platform](Docker%20and%20the%20RIS%20Compute1%20Platform.md)
> - [Docker Basics: Building, Tagging, & Pushing A Custom Docker Image](../Docker/Docker%20Basics_%20Building,%20Tagging,%20&%20Pushing%20A%20Custom%20Docker%20Image.md)

# Overview

This documentation will guide you on making sure you’re using the most appropriate CUDA version for your Docker image in regards to the Scientific Compute Platform.

# Using the Correct Version

Examples of appropriate base images:

```
nvidia/cuda:12.4.1-base-ubuntu22.04
nvidia/cuda:12.4.1-runtime-ubuntu22.04
ghcr.io/washu-it-ris/novnc:ubuntu22.04_cuda12.4_runtime
ghcr.io/washu-it-ris/novnc:ubuntu22.04_cuda12.4_devel
```

Nvidia has a lot of base images to develop from and can be found here: <https://hub.docker.com/r/nvidia/cuda/tags>

# Testing Your Image

Shown below are the steps to run a test job.

- Start up an interactive job with your Docker image.

  There is a test script in <https://github.com/WashU-IT-RIS/docker-osu-micro-benchmarks.git> for the OSU GPU test.
- Clone the repository.

```
git clone https://github.com/WashU-IT-RIS/docker-osu-micro-benchmarks.git
```

- Change directory to docker-osu-micro-benchmarks.

```
cd docker-osu-micro-benchmarks
```

- Run an OSU Benchmark GPU test.

  - Replace <test> with an OSU test that you want to run. For example, osu\_bw for OSU bandwidth test.
  - Replace <compute-group> with the compute group you are a member of.

```
QUEUE=subscription bin/osu-test-gpu.sh <test> -G <compute-group>
```
