
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Parabricks

- [Image Details](#image-details)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)
- [Additional Information](#additional-information)

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

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Image Details

- Docker image hosted at [nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1](http://nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1)
- Official documentation for [Parabricks version 4.0.0.](https://docs.nvidia.com/clara/parabricks/4.0.0/index.html)

# Getting Started

- Connect to compute client.

```
ssh washukey@compute1-client-1.ris.wustl.edu
```

- Prepare the computing environment before submitting a job.

```
# Use scratch file system for temp space
export SCRATCH1=/scratch1/fs1/${COMPUTE_ALLOCATION}

# Use Active storage for input and output data
export STORAGEN=/storageN/fs1/${STORAGE_ALLOCATION}/Active

# Use host level communications for the GPUs
export LSF_DOCKER_NETWORK=host

# Use debug flag when trying to figure out why your job failed to launch on the cluster
#export LSF_DOCKER_RUN_LOGLEVEL=DEBUG

# Use entry point since the parabricks container has other entrypoints but our cluster, by default, requires /bin/sh
export LSF_DOCKER_ENTRYPOINT=/bin/sh

# Create tmp dir
export TMP_DIR=${SCRATCH1}"/parabricks-tmp"
[ ! -d $TMP_DIR ] && mkdir $TMP_DIR
```

- Submit job. Basic commands for use:

```
bsub -n 16 -M 64GB -R 'gpuhost rusage[mem=64GB] span[hosts=1]' -q general -gpu "num=1:j_exclusive=yes" -a 'docker(nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1)' pbrun command options
```

# Known Issues

- Parabricks relies on available GPU(s) noted with `NVIDIA_VISIBLE_DEVICES` which defaults to ‘all’ regardless the quantity and

  device number of GPU(s) reserved at runtime. As such, there is a possibility the software will attempt to run on GPU(s) the job does not have access to. At this time it is advised to prepend `pbrun` with the following.

  ```
  for VAR in $(printenv | grep CUDA_VISIBLE_DEVICES); do export ${VAR/CUDA/NVIDIA}; done
  ```

# Additional Information

- Cores (`-n`) and memory (`-M` and `mem`) may need to be adjusted depending on the data set used.

  - 1 GPU server should have 64GB CPU RAM, at least 16 CPU threads.
  - 2 GPU server should have 100GB CPU RAM, at least 24 CPU threads.
  - 4 GPU server should have 196GB CPU RAM, at least 32 CPU threads.
- It is suggested to keep the GPUs at 4 and RAM at 196GB unless your data set is smaller than the 5GB test data set.
- There is diminishing returns on using more GPUs on small data sets.
- Replace `command` with any of the `pbrun` commands such as `fq2bam`, `bqsr`, `applybqsr`, or `haplotypecaller`.
- Please refer to official [Parabricks documentation](https://docs.nvidia.com/clara/parabricks/4.0.0/index.html) for additional direction.
