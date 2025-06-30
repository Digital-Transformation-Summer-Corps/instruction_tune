
[Compute2](../Compute2.md)

# Monitoring Jobs and Partitions/Queues

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Kill Running Jobs](#kill-running-jobs)
- [Check Job Status](#check-job-status)
- [Check Queues/Partitions](#check-queues-partitions)
- [Check Job Output](#check-job-output)
- [Query Slurm Accounting/Usage Data](#query-slurm-accounting-usage-data)

# Kill Running Jobs

Jobs can be killed by using [scancel](https://slurm.schedmd.com/scancel.html). Shown below are some example commands.

```
# Kill all PENDING job owned by sleong on partition general-short
scancel --state=PENDING --user=sleong --partition=general-short
# Kill the job with job id 1234
scancel 1234
# Send signal to the job with job id 1234 with signal USR1
scancel --signal=USR1 1234
```

# Check Job Status

To check Slurm job status, `squeue` and `scontrol` are used. Shown below are example commands.

```
# Show all jobs
squeue
# Show a user's jobs
squeue -u sleong
# Show a specific job 
squeue --job 1234
# Show detail of a job
scontrol show job=1234
```

# Check Queues/Partitions

The `sinfo` is used to check Slurm Queues/Partitions. Shown below is an example command and output.

```
[sleong@c2-login-001 ~]$ sinfo
PARTITION           AVAIL  TIMELIMIT  NODES  STATE NODELIST
consumption            up   infinite      0    n/a 
general*               up   infinite      5  inval c2-gpu-[017-018],c2-node-[084-086]
general*               up   infinite      2  down* c2-gpu-[010,012]
general*               up   infinite      1   drng c2-node-001
general*               up   infinite      2  drain c2-node-[002,004]
general*               up   infinite      2    mix c2-gpu-011,c2-node-003
general*               up   infinite      1  alloc c2-gpu-013
general*               up   infinite     90   idle c2-bigmem-001,c2-gpu-[001-009,014-015],c2-node-[005-078,080-083]
general-preemptable    up   infinite      0    n/a 
general-short          up      30:00      5  inval c2-gpu-[017-018],c2-node-[084-086]
general-short          up      30:00      2  down* c2-gpu-[010,012]
general-short          up      30:00      1   drng c2-node-001
general-short          up      30:00      2  drain c2-node-[002,004]
general-short          up      30:00      2    mix c2-gpu-011,c2-node-003
general-short          up      30:00      1  alloc c2-gpu-013
general-short          up      30:00     90   idle c2-bigmem-001,c2-gpu-[001-009,014-015],c2-node-[005-078,080-083]
subscription           up   infinite      0    n/a 
```

There is another Slurm command `sjstat` that is similar to the LSF `bqueues`. Shown below is an example command and output.

```
[sleong@c2-login-001 ~]$ sjstat -c
Scheduling pool data:
-------------------------------------------------------------
Pool        Memory  Cpus  Total Usable   Free  Other Traits  
-------------------------------------------------------------
consumpti       0Mb     0      0      0      0   
general*  4116480Mb    64      1      1      1  location=local 
general*  1024000Mb    64    101     94     90  location=local 
general*   773120Mb   256      1      1      1  location=local 
general-p       0Mb     0      0      0      0   
general-s 4116480Mb    64      1      1      1  location=local 
general-s 1024000Mb    64    101     94     90  location=local 
general-s  773120Mb   256      1      1      1  location=local 
subscript       0Mb     0      0      0      0   
```

# Check Job Output

There is a similar command to LSF `bpeek` in Slurm called `sattach`. You can use it to check for standard output and error of your jobs. Shown below is an example command.

```
# Usage: sattach [options] <jobid.stepid>
sattach 15568.0
```

# Query Slurm Accounting/Usage Data

If you want to query your usage and accounting information, Slurm provides a command `sacct` to query Slurm accounting data. Shown below are example commands. Please refer to [Slurm Workload Manager - sacct](https://slurm.schedmd.com/sacct.html) documentation for details.

```
# Show your usage
sacct
# Show your usage in details
sacct -l
```

A job usage data can be queried with the command `sstat`. Shown below is an example command.

```
sstat 1234
```
