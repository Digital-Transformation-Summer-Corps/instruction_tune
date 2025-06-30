
[Compute1](../Compute1.md)

# Intel® Compiler Base Tutorial

- [Overview](#overview)
- [Requirements](#requirements)
- [Multi-Stage Build](#multi-stage-build)
- [Interactive Command-Line Session](#interactive-command-line-session)
- [Single-Stage Build](#single-stage-build)

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

The purpose of this tutorial is to demonstrate usage of the Intel® Compiler Base image for use on the Scientific Compute Platform. Please refer to [Intel® Compiler Base](../Applications/Compute1%20Applications/RIS%20Developed%20Containers/Intel®%20Compiler%20Base.md) for more information on the Intel® Compiler Base image. In this tutorial, we will be compiling a sample MPI-enabled `Hello World` program. The tutorial uses material from [this page](https://mpitutorial.com/tutorials/mpi-hello-world/).

# Requirements

- Docker Desktop (<https://www.docker.com/products/docker-desktop> )
- Free Docker Hub Account ([https://hub.docker.com](https://hub.docker.com/))

# Multi-Stage Build

This section of the tutorial will demonstrate how to build a multi-stage Docker image for use on the Scientific Compute Platform. An MPI-enabled `Hello World` program is compiled using the Intel® Compiler Base image.

## MPI Hello World Code Example

Below is sample code for a simple MPI-enabled `Hello World` program.

```
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
          processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}
```

## Dockerfile for a Multi-Stage Build

Using your favorite text editor, save the above code to a file called `mpi_hello_world.c`. In the same folder as the `mpi_hello_world.c` file, create a file called `Dockerfile`. Add the following lines of code to the `Dockerfile` file:

```
# Begin Stage 1 with the base compiler image.
FROM ghcr.io/washu-it-ris/compiler-base:ubuntu22-mofed5.8-oneapi2025 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Additional dependencies for the Intel® MPI C compiler
RUN apt-get update && apt-get install -y gcc

# Change working directory
WORKDIR /opt/mpi_hello_world/src/

# Copy the source code to the working directory
COPY mpi_hello_world.c .

# Copy source code to a new location inside the container.
RUN cd /opt/mpi_hello_world/src/ && \
    . /opt/intel/oneapi/setvars.sh --force && \
    mpiicx -o mpi_hello_world mpi_hello_world.c && \
    cp -f mpi_hello_world /usr/local/bin

# Begin Stage 2 with a new base image.
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# Copy only the needed parts of Stage 1.
COPY --from=build /usr/local/bin/mpi_hello_world /usr/local/bin

# Add any additional runtime dependencies here
RUN apt-get update && apt-get install -y wget perl-base libnuma-dev \
    libgtk2.0 libatk1.0-0 libcairo2 gfortran tcsh libnl-3-dev \
    libmnl0 tcl tk \
    libusb-1.0-0-dev pciutils lsof ethtool libfuse2

# Set MOFED version, OS version and platform
ENV MOFED_VERSION=5.8-6.0.4.2
ENV OS_VERSION=ubuntu22.04
ENV PLATFORM=x86_64
RUN mkdir /tmp/mofed && \
    cd /tmp/mofed && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}/mlnxofedinstall \
      --user-space-only \
      --without-fw-update  \
      -q && \
    rm -rf /tmp/mofed

# Install Intel OneAPI
RUN apt-get update -y && apt-get install -y gpg-agent wget curl software-properties-common ca-certificates
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --yes --dearmor |  tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |  tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update -y && apt-get install -y intel-oneapi-hpc-toolkit
```

## Building and Pushing the Docker Image

Please see [Intel® Compiler Base](../Applications/Compute1%20Applications/RIS%20Developed%20Containers/Intel®%20Compiler%20Base.md) for more information on building and pushing the Docker image to Docker Hub.

# Interactive Command-Line Session

## Job Submission

The following command can be used to submit an interactive command-line job to run the MPI-enabled `Hello World` program. Make sure to replace `docker/image` with the name of the Docker image you created.

```
LSF_DOCKER_NETWORK=host LSF_DOCKER_IPC=host bsub -n 20 -Is -q general-interactive -R "affinity[core(1):distribute=pack] span[ptile=4]" -a "docker(docker/image)" /bin/bash
```

The job submission command uses environment variables required for parallel computing on RIS. The job submission is also requesting 20 vCPUs spread across 5 exec nodes. Please see the [Parallel Computing](Parallel%20Computing.md) for more information.

## Running the MPI-enabled `Hello World` program

Run the MPI-enabled `Hello World` program using 20 vCPUs with the following command:

```
mpirun -np 20 /usr/local/bin/mpi_hello_world
```

The following output should be displayed after the image has finished downloading on all exec nodes:

```
Hello world from processor compute1-exec-98.ris.wustl.edu, rank 12 out of 20 processors
Hello world from processor compute1-exec-98.ris.wustl.edu, rank 13 out of 20 processors
Hello world from processor compute1-exec-98.ris.wustl.edu, rank 14 out of 20 processors
Hello world from processor compute1-exec-98.ris.wustl.edu, rank 15 out of 20 processors
Hello world from processor compute1-exec-6.ris.wustl.edu, rank 5 out of 20 processors
Hello world from processor compute1-exec-6.ris.wustl.edu, rank 6 out of 20 processors
Hello world from processor compute1-exec-6.ris.wustl.edu, rank 4 out of 20 processors
Hello world from processor compute1-exec-6.ris.wustl.edu, rank 7 out of 20 processors
Hello world from processor compute1-exec-5.ris.wustl.edu, rank 1 out of 20 processors
Hello world from processor compute1-exec-218.ris.wustl.edu, rank 8 out of 20 processors
Hello world from processor compute1-exec-218.ris.wustl.edu, rank 9 out of 20 processors
Hello world from processor compute1-exec-5.ris.wustl.edu, rank 3 out of 20 processors
Hello world from processor compute1-exec-218.ris.wustl.edu, rank 10 out of 20 processors
Hello world from processor compute1-exec-5.ris.wustl.edu, rank 0 out of 20 processors
Hello world from processor compute1-exec-218.ris.wustl.edu, rank 11 out of 20 processors
Hello world from processor compute1-exec-5.ris.wustl.edu, rank 2 out of 20 processors
Hello world from processor compute1-exec-217.ris.wustl.edu, rank 19 out of 20 processors
Hello world from processor compute1-exec-217.ris.wustl.edu, rank 16 out of 20 processors
Hello world from processor compute1-exec-217.ris.wustl.edu, rank 17 out of 20 processors
Hello world from processor compute1-exec-217.ris.wustl.edu, rank 18 out of 20 processors
```

# Single-Stage Build

Should you require a single-stage build, the following Dockerfile can be used. Please be aware that this method results in a larger Docker image which may cause increased computing time/resources/cost. This method also caches the source code in build layers resulting in public exposure, which may be unwanted. When possible, it is advised to use the multi-stage build method.

```
# Begin Stage 1 with the base compiler image.
FROM ghcr.io/washu-it-ris/compiler-base:ubuntu22-mofed5.8-oneapi2025 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Additional dependencies for the Intel® MPI C compiler
RUN apt-get update && apt-get install -y gcc wget perl-base libnuma-dev \
    libgtk2.0 libatk1.0-0 libcairo2 gfortran tcsh libnl-3-dev \
    libmnl0 tcl tk \
    libusb-1.0-0-dev pciutils lsof ethtool libfuse2

# Change working directory
WORKDIR /opt/mpi_hello_world/src/

# Copy the source code to the working directory
COPY mpi_hello_world.c .

# Copy source code to a new location inside the container.
RUN cd /opt/mpi_hello_world/src/ && \
    . /opt/intel/oneapi/setvars.sh --force && \
    mpiicx -o mpi_hello_world mpi_hello_world.c && \
    cp -f mpi_hello_world /usr/local/bin

# Set MOFED version, OS version and platform
ENV MOFED_VERSION=5.8-6.0.4.2
ENV OS_VERSION=ubuntu22.04
ENV PLATFORM=x86_64
RUN mkdir /tmp/mofed && \
    cd /tmp/mofed && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}/mlnxofedinstall \
      --user-space-only \
      --without-fw-update  \
      -q --force && \
    rm -rf /tmp/mofed

# Install Intel OneAPI
RUN apt-get update -y && apt-get install -y gpg-agent wget curl software-properties-common ca-certificates
RUN wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --yes --dearmor |  tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |  tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update -y && apt-get install -y intel-oneapi-hpc-toolkit
```
