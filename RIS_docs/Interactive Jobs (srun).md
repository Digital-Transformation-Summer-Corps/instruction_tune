
[Compute2](../Compute2.md)

# Interactive Jobs (srun)

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Default Usage](#default-usage)
- [Partition/Queue Options](#partition-queue-options)
- [CPU Options](#cpu-options)
- [GPU Options](#gpu-options)
- [Memory/RAM Options](#memory-ram-options)
- [Constraint Options](#constraint-options)
- [Specific Host Options](#specific-host-options)
- [Job Priority Options](#job-priority-options)
- [Additional Information](#additional-information)

# Default Usage

```
srun [OPTIONS(0)... [executable(0) [args(0)...]]] [ : [OPTIONS(N)...]] executable(N) [args(N)...]
```

Examples of basic interactive jobs:

```
# A bare metal job
[washukey@c2-login-001 ~]$ srun hostname
c2-node-003

# A container job
[washukey@c2-login-001 ~]$ srun --container-image=ubuntu hostname
pyxis: importing docker image: ubuntu
pyxis: imported docker image: ubuntu
c2-node-003
```

Examples of interactive jobs with a terminal prompt:

```
# A bare metal job
[washukey@c2-login-001 ~]$ srun --pty /bin/bash
[washukey@c2-node-003 ~]$ 

# A container job
[washukey@c2-login-001 ~]$ srun --pty --container-image=ubuntu /bin/bash
pyxis: importing docker image: ubuntu
pyxis: imported docker image: ubuntu
washukey@c2-node-003:/$
```

# Partition/Queue Options

Below are examples of requesting a partition or queue.

### Request job for the `general` partition/queue

```
srun --partition=general <command>
```

# CPU Options

Below are examples of requesting CPU resources beyond the default.

### Request job with 16 tasks

```
srun -n 16 <command>
```

### Request job with 2 nodes and 4 tasks per nodes

```
srun -N 2 --ntasks-per-node=4 <command>
```

# GPU Options

Below are examples for requesting GPU resources.

### Request a job with 2 GPUs

```
srun --gpus=2 <command>
```

### Further Options

```
GPU scheduling options:
      --cpus-per-gpu=n        number of CPUs required per allocated GPU
  -G, --gpus=n                count of GPUs required for the job
      --gpu-bind=...          task to gpu binding options
      --gpu-freq=...          frequency and voltage of GPUs
      --gpus-per-node=n       number of GPUs required per allocated node
      --gpus-per-socket=n     number of GPUs required per allocated socket
      --gpus-per-task=n       number of GPUs required per spawned task
      --mem-per-gpu=n         real memory required per allocated GPU
```

# Memory/RAM Options

Below are examples for requesting memory/RAM resources.

### Request a job with 100GB memory/RAM

```
srun --mem=100G <command>
```

### Further Options

```
      --mem=MB                minimum amount of real memory
      --mem-per-cpu=MB        maximum amount of real memory per allocated
                              cpu required by the job.
                              --mem >= --mem-per-cpu if --mem is specified.

```

# Constraint Options

Slurm provides `--constraint` or `-C` to specify required resources that nodes must meet for jobs to be scheduled.

Below is the table describes each feature in the Compute2 cluster:

|  **Feature**    |  **Description**                       |
|:----------------|:---------------------------------------|
| intel           | Intel CPU processors                   |
| amd             | AMD CPU processors                     |
| sapphirerapids  | Intel’s Sapphire Rapids processor line |
| znver3          | AMD’s Zen version 3 processor line     |

### Request a job with AMD CPUs

```
srun --constraint=amd <command>
```

# Specific Host Options

> [!IMPORTANT]
> A host might not be immediately accessible due to resource constraints or access restriction. Please check the nodes first before submitting to job specific hosts.

Specific hosts can be specified when submitting a job using the `-w` flag.

### Request job using c2-node-001 host

```
srun -w c2-node-001 <command>
```

# Job Priority Options

The `--priority` flag sets a numeric priority to order the executions of jobs.

```
--priority=value        set the priority of the job to value
```

### Request a job setting the priority to 99

```
srun --priority=99 <command>
```

> [!IMPORTANT]
> A job’s priority can be modified while in the `PENDING` state by using `scontrol`.

### Change the priority of job 1234 to 99

```
scontrol update job=1234 priority=99
```

# Additional Information

The options listed are not all inclusive. If you are looking for something, and cannot find it here, please checkout the [Slurm documentation on srun](https://slurm.schedmd.com/srun.html) for more information.
