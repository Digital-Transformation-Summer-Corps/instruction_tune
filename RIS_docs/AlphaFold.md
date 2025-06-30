
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# AlphaFold

- [Software Included](#software-included)
- [Getting Started](#getting-started)
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

# Software Included

- AlphaFold v2.2.0 (<https://github.com/deepmind/alphafold> )

# Getting Started

- Connect to compute client.

```
ssh washukey@compute1-client-1.ris.wustl.edu
```

- Prepare the computing environment before submitting an AlphaFold job.

```
  # Set the AlphaFold base directory
  export ALPHAFOLD_BASE_DIR=/app/alphafold

  # Use the scratch file system for temp space
  export SCRATCH1=/scratch1/fs1/${COMPUTE_ALLOCATION}

  # Use your Active storage for input and output data
  export STORAGE1=/storageN/fs1/${STORAGE_ALLOCATION}/Active

  # Mount scratch, Active storage, home directory and AlphaFold database reference files
  export LSF_DOCKER_VOLUMES="/scratch1/fs1/ris/references/alphafold_db:/scratch1/fs1/ris/references/alphafold_db $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1 $HOME:$HOME"

  # Update $PATH with folders containing AlphaFold, CUDA, and conda executables
  export PATH="/usr/local/cuda/bin/:/opt/conda/bin:/app/alphafold:$PATH"

  # Use the debug flag when trying to figure out why your job failed to launch on the cluster
  #export LSF_DOCKER_RUN_LOGLEVEL=DEBUG


- The following run example implements AlphaFold's suggested system requirements for database preset ``reduced_dbs``.
```

```
bsub -q general -n 8 -M 8GB -R "gpuhost rusage[mem=8GB] span[hosts=1]" -gpu 'num=1' -a "docker(gcr.io/ris-registry-shared/alphafold:2.2.0)" python3 /app/alphafold/run_alphafold.py --output_dir /path/to/output/folder --model_preset monomer --fasta_paths /path/to/input/protein_sequence.fa --max_template_date 2021-08-18 --db_preset reduced_dbs
```

- AlphaFold can run by default on both V100 and A100 GPU architectures. Modify the `-gpu` argument to specify the GPU architecture.

```
-gpu 'num=1:gmodel=<gpu_model>'
```

- A list of GPU models can be found [Job Execution Examples](../../../Compute1/Job%20Execution%20Examples.md).
- Jobs can be managed using [Job Execution Examples](../../../Compute1/Job%20Execution%20Examples.md). Job groups are a way to submit a large number of jobs at once.

# Additional Information

Please refer to official [AlphaFold documentation](https://github.com/deepmind/alphafold/tree/v2.2.0) for direction on setting up run options, expected output, example runs, etc.
