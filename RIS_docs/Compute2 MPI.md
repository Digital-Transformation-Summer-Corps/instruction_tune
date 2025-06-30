
[Compute2](../Compute2.md)

# Compute2 MPI

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [MPI Jobs](#mpi-jobs)
- [OpenMPI](#openmpi)
- [Intel MPI](#intel-mpi)

# MPI Jobs

OpenMPI and Intel MPI jobs are supported with or without containers on the Compute2 cluster. The Compute2 Slurm cluster supports the Process Management Interface: PMIx and PMI2. For MPI jobs, special care is needed for each type of MPI jobs.

# OpenMPI

When running OpenMPI jobs, PMIX option is needed to run MPI jobs. The `--mpi=pmix` option can only be specified on `srun` command. When submitting a batch job with `sbatch`, you will prefix `mpirun` command with `srun --mpi=pmix`. Shown below is an example batch job that will run `mympiprogram`.

1. Create a script called `mympiscript.sh` with the content below. Make sure the `mympiscript.sh` is executable.

   ```
   #!/bin/bash
   #SBATCH -N 2
   #SBATCH --mem=10G
   #SBATCH --ntasks-per-node=16
   srun --mpi=pmix mpirun ./mympiprogram
   ```
2. Run a batch job. Shown below is an example command.

   ```
   sbatch ./mympiscript.sh
   ```

# Intel MPI

Intel MPI does not support PMIX. It can only use the `--mpi=pmi2`. It is the same as the OpenMPI example above that the `--mpi=pmi2` can only be specified on `srun` command. When submitting a batch job with `sbatch`, you will prefix `mpirun` command with `srun --mpi=pmi2`. Since it is using `pmi2`, you need to tell Intel MPI where to find the `pmi2` library by setting the `I_MPI_PMI_LIBRARY` environment variable to `/cm/shared/apps/slurm/23.02.5/lib64/libpmi2.so`. Shown below is an example batch job that will run `mympiprogram`.

1. Create a script called `mympiscript.sh` with the content below. Make sure the `mympiscript.sh` is executable.

   ```
   #!/bin/bash
   #SBATCH -N 2
   #SBATCH --mem=10G
   #SBATCH --ntasks-per-node=16
   export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/23.02.5/lib64/libpmi2.so
   srun --mpi=pmi2 mpirun ./mympiprogram
   ```
2. Run a batch job. Shown below is an example command.

   ```
   sbatch ./mympiscript.sh
   ```
