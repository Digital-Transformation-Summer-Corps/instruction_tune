
[Compute1](../Compute1.md)

# Using OFED in Docker Images

- [Overview](#overview)
- [Installing the Correct Version](#installing-the-correct-version)
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

This documentation will guide you on making sure you’re using the most appropriate OFED version for your Docker image in regards to the Scientific Compute Platform.

# Installing the Correct Version

Shown below is an example of OFED 5.8-4.1.5.0 driver Dockerfile instructions for RedHat 8.9.

```
ENV MOFED_VERSION 5.8-4.1.5.0
ENV OS_VERSION rhel8.9
ENV PLATFORM x86_64
RUN cd /tmp/ && yum install -y pciutils numactl-libs gtk2 atk cairo gcc-gfortran tcsh lsof libnl3 libmnl ethtool tcl tk perl make libusbx fuse-libs && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update -q  --distro rhel8.9 && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz && \
    yum clean all
```

This also pertains to the `Ubuntu 22.04` with different code snippets but same version of `OFED 5.8-4.1.5.0`.

```
ENV MOFED_VERSION 5.8-4.1.5.0
ENV OS_VERSION ubuntu22.04
ENV PLATFORM x86_64
RUN cd /tmp/ && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y dkms wget && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update -q --distro ${OS_VERSION} && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz && \
    apt-get clean
```

Once you have the correct OFED version installation code in your Dockerfile, you can build and push the image as you normally would.

# Testing Your Image

Shown below are the steps to run a test job.

- Create a bsub file called test.bsub as shown below. Please replace <Docker image tag> with your Docker image tag and <MPI program>.

```
#BSUB -q subscription
#BSUB -R "span[ptile=1]"
#BSUB -a "docker(<Docker image tag>)"
#BSUB -G compute-ris
#BSUB -oo lsf-%J.log

mpirun -np $NP <MPI program>
```

- Run your test. Shown below is an example command. Please replace <Number of processes> with number of exec nodes to run the test.

```
export NP=<Number of processes> && \
LSF_DOCKER_NETWORK=host \
LSF_DOCKER_IPC=host \
LSF_DOCKER_SHM_SIZE=20G \
bsub -n $NP < test.bsub
```

- There is a test script in <https://github.com/WashU-IT-RIS/docker-osu-micro-benchmarks.git> . Shown below are the instructions for OSU Benchmark test.

  - Clone the repository.

  ```
  git clone https://github.com/WashU-IT-RIS/docker-osu-micro-benchmarks.git
  ```

  - Change directory to docker-osu-micro-benchmarks.

  ```
  cd docker-osu-micro-benchmarks
  ```

  - Run an OSU Benchmark test.

    - Replace <test> with an OSU test that you want to run. For example, osu\_bw for OSU bandwidth test.
    - Replace <compute-group> with the compute group you are a member of.

  ```
  QUEUE=subscription bin/osu-test.sh <test> -G <compute-group>
  ```
