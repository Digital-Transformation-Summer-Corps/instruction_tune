
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Intel® Compiler Base

- [Software Included](#software-included)
- [Overview](#overview)
- [Interactive Command-Line Session](#interactive-command-line-session)
- [Multi-Stage Docker Image Build](#multi-stage-docker-image-build)
- [Single-Stage Docker Image Build](#single-stage-docker-image-build)
- [Building and Pushing the Docker image](#building-and-pushing-the-docker-image)
- [Intel® Base Compiler Tutorial](#intel-base-compiler-tutorial)

> [!IMPORTANT]
> Compute Resources
>
> - Have questions or need help with compute, including activation or issues? Follow [this link.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43)
> - [RIS Services Policies](../../../RIS%20Services%20Policies.md)

> [!IMPORTANT]
> Docker Usage
>
> - The information on this page assumes that you have a knowledge base of using Docker to create images and push them to a repository for use. If you need to review that information, please see the links below.
> - [Docker and the RIS Compute1 Platform](../../../Compute1/Docker%20and%20the%20RIS%20Compute1%20Platform.md)
> - [Docker Basics: Building, Tagging, & Pushing A Custom Docker Image](../../../Docker/Docker%20Basics_%20Building,%20Tagging,%20&%20Pushing%20A%20Custom%20Docker%20Image.md)

# Software Included

- Intel® oneAPI HPC Toolkit 2021.1.1 (<https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html> )

# Overview

From the [Intel® oneAPI HPC Toolkit Docker Hub](https://hub.docker.com/r/intel/oneapi-hpckit),

*The Intel oneAPI HPC Toolkit delivers what developers need to build, analyze, optimize, and scale high-performance computing (HPC) applications with the latest techniques in vectorization, multithreading, multi-node parallelization, and memory optimization.*

For a complete list of libraries included in the Intel® oneAPI HPC Toolkit, please visit the [Intel® oneAPI HPC Toolkit Documentation](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top.html).

# Interactive Command-Line Session

To use the Intel® compilers in an interactive command-line session, follow the steps below.

1. Submit an interactive job.

```
LSF_DOCKER_VOLUMES="/scratch1/fs1/ris/application/intel/oneapi/:/opt/intel/oneapi" \
bsub -Is -q general-interactive -a 'docker(ghcr.io/washu-it-ris/compiler-base:oneapi2021.1.1_centos7)' \
/bin/bash
```

2. Set up the Intel® oneAPI environment.

```
. /opt/intel/oneapi/setvars.sh
```

3. Compile code.

   - Please refer to the [Intel® documentation](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-setup/using-the-command-line/invoking-the-compiler.html) for more information on how to compile code.

# Multi-Stage Docker Image Build

## Compile, Keep Only Binaries

A multi-stage build leverages the compilers in the Intel® oneAPI HPC Toolkit Docker image and copies the compiled binaries and runtime dependencies to a new base image. This method results in a smaller Docker image, reducing computing time/resources/cost, and allows withholding source code from public consumption.

For more information on Docker multi-stage builds, please see the [Intel® Compiler Base Tutorial](../../../Compute1/Intel®%20Compiler%20Base%20Tutorial.md).

## Sample Multi-Stage Dockerfile

```
# Begin Stage 1 with the base compiler image.
FROM ghcr.io/washu-it-ris/compiler-base:oneapi2021.1.1_centos7 as build

# Add any additional build dependencies here.

# copy source code to a new location inside the container.
COPY /path/to/source/code /opt/app_name/src

# Change directory to location of source code,
# set up the Intel environment,
# compile,
# copy binary to standard location.
RUN cd /opt/app_name/src/ && \
    . /opt/intel/oneapi/setvars.sh && \
    make && \
    cp -f example.binary /usr/local/bin

# Begin Stage 2 with a new base image.
FROM centos:7.9.2009

# Copy only the needed parts of Stage 1.
COPY --from=build /usr/local/bin/example.binary /usr/local/bin
COPY --from=build /usr/local/lib /usr/local/lib
COPY --from=build /usr/local/include /usr/local/include

# Add any additional runtime dependencies here.

# Set up MLNX_OFED driver.
ENV MOFED_VERSION 5.8-4.1.5.0
ENV OS_VERSION rhel7.9
ENV PLATFORM x86_64
RUN wget -q http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update -q  --distro ${OS_VERSION} && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz
```

# Single-Stage Docker Image Build

## Compile, Compile, Keep Source Code and Binaries

A single-stage build leverages the compilers in the Intel® oneAPI HPC Toolkit Docker image and keeps the compiled binaries, runtime dependencies and source code in the resulting image. This method results in a larger Docker image which may cause increased computing time/resources/cost. This method also caches the source code in build layers resulting in public exposure, which may be unwanted.

## Sample Single-Stage Dockerfile

```
FROM ghcr.io/washu-it-ris/compiler-base:oneapi2021.1.1_centos7

# Add any additional build dependencies here.

# copy source code to a new location inside the container
COPY /path/to/source/code /opt/app_name/src

# Change directory to location of source code,
# set up the Intel environment,
# compile.
RUN cd /opt/app_name/src/ && \
    . /opt/intel/oneapi/setvars.sh && \
    make
```

# Building and Pushing the Docker image

To build and push a Docker image using one of the above methods, please refer to the our existing documentation for guidance.

- [Docker and the RIS Compute1 Platform](../../../Compute1/Docker%20and%20the%20RIS%20Compute1%20Platform.md)
- [Docker Basics: Building, Tagging, & Pushing A Custom Docker Image](../../../Docker/Docker%20Basics_%20Building,%20Tagging,%20&%20Pushing%20A%20Custom%20Docker%20Image.md)
- [Docker Tutorial](../../../Docker/Docker%20Tutorial.md)

# Intel® Base Compiler Tutorial

- [Intel® Compiler Base Tutorial](../../../Compute1/Intel®%20Compiler%20Base%20Tutorial.md)

Please see our [Intel® Compiler Base Tutorial](../../../Compute1/Intel®%20Compiler%20Base%20Tutorial.md) to learn how to leverage the Intel® Base Compiler Docker image on the RIS Scientific Compute Platform.
