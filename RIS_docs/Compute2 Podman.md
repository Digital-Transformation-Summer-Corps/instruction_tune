
[Applications](../../Applications.md) > [Compute2 Applications](../Compute2%20Applications.md)

# Compute2 Podman

- [Overview](#overview)
- [Quick-Start](#quick-start)

# Overview

[Podman](https://podman.io/docs) is the preferred, best practies, way to build [OCI containers](https://opencontainers.org/) (Docker) on Compute2.

Podman is very similar to Docker as far as an end user is concerned. There are some edge cases with respect to `compose.yaml` support and multi-arch builds.

We are choosing to run all Podman builds within a SLURM job. This is to help with cleanup of the underlying QEMU VM that we need when executing Podman commands. **Killing the SLURM job will stop the VM.**

# Quick-Start

## 1. Load the Relevant Modules

```
module load ris podman slurm
```

## 2. Create `temp` Directories to Hold VM and Container Data

```
mkdir -p /scratch2/fs1/allocation_name/podman/runtime
```

> [!IMPORTANT]
> - `allocation_name` is the name of the scratch allocation. E.g. `ris`

## 3. Start an Interactive `slurm` Job

```
XDG_CONFIG_HOME=/scratch2/fs1/allocation_name/podman \         # defualt: $HOME/.config 
XDG_DATA_HOME=/scratch2/fs1/allocation_name/podman \           # default: $HOME/.local/share 
XDG_RUNTIME_DIR=/scratch2/fs1/allocation_name/podman/runtime \ # default: /run/user/$UID 
srun --pty /bin/bash
```

> [!IMPORTANT]
> - You must set `XDG_` vars noted above or your `home2` will fill up!
> - Do not `export` these variables unless you are certain you know what you’re doing.

## 4. Start a [Podman Machine](https://docs.podman.io/en/stable/markdown/podman-machine.1.html) Inside the SLURM Job

```
podman machine init
podman machine start
```

## 5. Create a Basic “Dockerfile” in the Working Directory

```
FROM rockylinux:9
# Install EPEL and CRB Repositories
RUN dnf install epel-release -y && \
    crb enable
# Install something fun
RUN dnf install -y sl
```

> [!IMPORTANT]
> `Dockerfile` and `Containerfile` are both valid “default” filenames.

## 6. Perform a `podman build` Just Like Docker

- The syntax is largely the same as Docker.
- The -f option is not needed if using a default file name as indicated above.

```
podman build -t build-test -f ./Dockerfile .
```

> [!IMPORTANT]
> - Users can interact with any directory via Podman run or build (`storage2`, `scratch2`, `storage1`, etc), assuming that location is available in the shell session on the Compute2 node.
> - Don’t worry about mounting “extra” locations to the VM.

## 7. Confirm the Container Works

```
podman run -it --rm build-test sl
```

![image-20250604-040459.png](../../../attachments/415d72a1-8581-4fc4-b6c2-1da908d4e405.png)
