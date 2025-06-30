
[Compute2](../Compute2.md)

# Compute2 General Guidelines

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Basics](#basics)

# Basics

Slurm puts a lot of hardware at your fingertips, but it must be used wisely so you don’t affect others’ work. This page contains guidelines for using Slurm.

- Don’t submit 1000 jobs until you’ve seen 1 job finish successfully.
- Job submission has overhead. Avoid submitting jobs that complete in just seconds. If you can’t avoid it, look into job arrays.
- Limit the total number of files you place in a single directory. On the order of thousands at most. Normal filesystem operations become unwieldy with directories of millions of files.
- Every Slurm jobs gets its own temporary directory that gets cleaned up for you! Do your work there, if possible. The path is: `/tmp/`.
- Don’t monopolize queues with large numbers of long-running (multi-day) jobs. Use QOS to limit your running jobs if necessary.
- Use `srun` for [Interactive Jobs (srun)](Interactive%20Jobs%20(srun).md) and `sbatch` for [Batch Jobs (sbatch)](Batch%20Jobs%20(sbatch).md).
- Don’t rely on the host environment to develop or install your software.

  - Create your own environment using container technology, or
  - Encapsulate all your dependencies
