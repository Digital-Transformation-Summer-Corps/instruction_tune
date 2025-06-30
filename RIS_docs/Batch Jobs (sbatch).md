
[Compute2](../Compute2.md)

# Batch Jobs (sbatch)

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Default Usage](#default-usage)
- [Partition/Queue Options](#partition-queue-options)
- [CPU Options](#cpu-options)
- [Specific Host Options](#specific-host-options)
- [GPU Options](#gpu-options)
- [Memory/RAM Options](#memory-ram-options)
- [Job Dependency Options](#job-dependency-options)
- [Job Array Options](#job-array-options)
- [Constraint Options](#constraint-options)
- [Job Priority Options](#job-priority-options)
- [Additional Information](#additional-information)

# Default Usage

```
sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]
```

Batch or `sbatch` jobs require a script to submit. It is recommended that options used be put into the script. Each option should be a separate line and start with `#SBATCH`

Example of a basic sbatch script (myscript.sh):

```
#!/bin/bash
#SBATCH -p general

<command>
```

Example of submitting an sbatch job:

```
sbatch myscript.sh
```

# Partition/Queue Options

Below are examples of requesting a partition or queue.

### Request job for the `general` partition/queue

```
#SBATCH --partition=general
```

# CPU Options

Below are examples of requesting CPU resources beyond the default.

### Request job with 16 tasks

```
#SBATCH -n 16
```

### Request job with 2 nodes and 4 tasks per nodes

```
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
```

# Specific Host Options

> [!IMPORTANT]
> A host might not be immediately accessible due to resource constraints or access restriction. Please check the nodes first before submitting to job specific hosts.

Specific hosts can be specified when submitting a job using the `-w` flag.

### Request job using c2-node-001 host

```
#SBATCH -w c2-node-001
```

# GPU Options

Below are examples for requesting GPU resources.

### Request a job with 2 GPUs

```
#SBATCH --gpus=2
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
#SBATCH --mem=100G
```

### Further Options

```
      --mem=MB                minimum amount of real memory
      --mem-per-cpu=MB        maximum amount of real memory per allocated
                              cpu required by the job.
                              --mem >= --mem-per-cpu if --mem is specified.

```

# Job Dependency Options

Below are examples of using dependencies when submitting `sbatch` jobs.

### Request a job that runs after job 1234 finishes successfully

```
#SBATCH --dependency=afterok:1234
```

### Further Options

```
-d, --dependency=<dependency_list>
    Defer the start of this job until the specified dependencies have been satisfied. Once a dependency is satisfied, it is removed from the job. <dependency_list> is of the form <type:job_id[:job_id][,type:job_id[:job_id]]> or <type:job_id[:job_id][?type:job_id[:job_id]]>. All dependencies must be satisfied if the "," separator is used. Any dependency may be satisfied if the "?" separator is used. Only one separator may be used. For instance:
    -d afterok:20:21,afterany:23
    means that the job can run only after a 0 return code of jobs 20 and 21 AND the completion of job 23. However:
    -d afterok:20:21?afterany:23
    means that any of the conditions (afterok:20 OR afterok:21 OR afterany:23) will be enough to release the job. Many jobs can share the same dependency and these jobs may even belong to different users. The value may be changed after job submission using the scontrol command. Dependencies on remote jobs are allowed in a federation. Once a job dependency fails due to the termination state of a preceding job, the dependent job will never be run, even if the preceding job is requeued and has a different termination state in a subsequent execution. 
        after:job_id[[+time][:jobid[+time]...]]
            After the specified jobs start or are cancelled and 'time' in minutes from job start or cancellation happens, this job can begin execution. If no 'time' is given then there is no delay after start or cancellation. 
        afterany:job_id[:jobid...]
            This job can begin execution after the specified jobs have terminated. This is the default dependency type. 
        afterburstbuffer:job_id[:jobid...]
            This job can begin execution after the specified jobs have terminated and any associated burst buffer stage out operations have completed. 
        aftercorr:job_id[:jobid...]
            A task of this job array can begin execution after the corresponding task ID in the specified job has completed successfully (ran to completion with an exit code of zero). 
        afternotok:job_id[:jobid...]
            This job can begin execution after the specified jobs have terminated in some failed state (non-zero exit code, node failure, timed out, etc). This job must be submitted while the specified job is still active or within MinJobAge seconds after the specified job has ended. 
        afterok:job_id[:jobid...]
            This job can begin execution after the specified jobs have successfully executed (ran to completion with an exit code of zero). This job must be submitted while the specified job is still active or within MinJobAge seconds after the specified job has ended. 
        singleton
            This job can begin execution after any previously launched jobs sharing the same job name and user have terminated. In other words, only one job by that name and owned by that user can be running or suspended at any point in time. In a federation, a singleton dependency must be fulfilled on all clusters unless DependencyParameters=disable_remote_singleton is used in slurm.conf. 

```

# Job Array Options

When submitting a lot of similar jobs, it is recommended to use a job array. This allows jobs to be submitted and managed as a single entity.

Below are examples using job arrays.

### Request a job array with indexes between 0 and 31

```
#SBATCH --array=0-31
```

### Request a job array with indexes of 1, 3, 5, and 7

```
#SBATCH --array=1,3,5,7
```

### Request a job array with indexes between 1 and 7 with a step size of 2

```
#SBATCH --array=1-7:2
```

Further information about job arrays can be found here: [Slurm Workload Manager - Job Array Support](https://slurm.schedmd.com/job_array.html)

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
#SBATCH --constraint=amd
```

# Job Priority Options

The `--priority` flag sets a numeric priority to order the executions of jobs.

```
--priority=value        set the priority of the job to value
```

### Request a job setting the priority to 99

```
#SBATCH --priority=99
```

> [!IMPORTANT]
> A job’s priority can be modified while in the `PENDING` state by using `scontrol`.

### Change the priority of job 1234 to 99

```
scontrol update job=1234 priority=99
```

# Additional Information

The options listed are not all inclusive. If you are looking for something, and cannot find it here, please checkout the [Slurm documentation on sbatch](https://slurm.schedmd.com/sbatch.html) for more information.
