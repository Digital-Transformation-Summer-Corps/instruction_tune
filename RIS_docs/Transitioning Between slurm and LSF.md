
[Compute2](../Compute2.md)

# Transitioning Between slurm and LSF

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Overview](#overview)
- [Comparible Commands](#comparible-commands)
- [Job Scripts](#job-scripts)

# Overview

- This page discusses the similarities between slurm and the LSF system when it comes to managing and submitting jobs.
- This page won’t cover everything about LSF but is designed to be a stepping stone to go from when transitioning to the Compute Platform.

# Comparible Commands

- The following list is slurm commands and their equivalent within LSF.

  - `sbatch` - `bsub`
  - `squeue` - `bjobs`
  - `scancel` - `bkill`
  - `sinfo` - `bhosts`
- More information about job commands can be found [in our job execution examples documentation](https://docs.ris.wustl.edu/doc/compute/recipes/job-execution-examples.html#job-execution-examples).

# Job Scripts

- Most of the documentation and use of the Compute Platform assumes submitting jobs directly with `bsub`.
- Jobs can also be submitted via job scripts just like in slurm.
- An example slurm job script might look like the following.

```
#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --output=results.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

command
```

- While an example job script in LSF would look like the following.

```
#!/bin/bash
#BSUB  -n 1
#BSUB -q submission_queue
#BSUB -G my_group
#BSUB -M 8000000
#BSUB -oo ${HOME}/path/to/log_file
#BSUB -a 'docker(container_name:container_tag)'

commands
```

- Users can create job scripts for LSF just like they would for slurm.
- On Compute1 the `-a` option must be included to specify which Docker image the job is using.
- Unlike a more traditional HPC system the Compute1 makes use of Docker images instead of modules while Compute2 has modules and the ability to use containers.
- You can read more about Docker and it’s association with Compute1 at the following links.

  - [Docker and the RIS Compute Service](https://docs.ris.wustl.edu/doc/compute/recipes/docker-on-compute.html#docker-on-compute)
  - [Docker Basics: Building, Tagging, & Pushing A Custom Docker Image](https://docs.ris.wustl.edu/doc/compute/workshops/ris-docker-basics.html#ris-docker-basics)
  - [RIS Docker Workshop](https://docs.ris.wustl.edu/doc/compute/workshops/ris-docker-workshop.html#ris-docker-workshop)
