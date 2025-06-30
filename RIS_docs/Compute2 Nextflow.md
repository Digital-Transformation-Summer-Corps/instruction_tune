
[Applications](../../Applications.md) > [Compute2 Applications](../Compute2%20Applications.md)

# Compute2 Nextflow

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Quick-Start](#quick-start)
- [Additional Information](#additional-information)

# Quick-Start

## 1. Load relevant modules

```
module load ris nextflow apptainer
```

## 2. Obtain the default `nextflow.config` file for use on Compute2.

```
wget https://github.com/WashU-IT-RIS/c2-ris-app-config-templates/blob/main/nextflow/nextflow.config
```

- This file can be placed anywhere you like with whatever name you like provided you use `-c /path/to/your.config` when executing Nextflow.
- The default location is in `$HOME/.config/nextflow/` within a file named `nextflow.config`
- Create a temporary output directory for Nextflow.

  ```
  mkdir -p /scratch2/fs1/allocation_name/nextflow/output
  ```

> [!IMPORTANT]
> - This can be other locations like an RIS Storage platform.
> - `scratch2` is preferred (and faster) if there is access available.
> - `allocation_name` is the name of the scratch allocation. E.g. `ris`

## 3. Execute a test pipeline

> [!IMPORTANT]
> Make sure that Nextflow commands are run as part of a job and are not run on the login clients.

```
nextflow run \
  nf-core/rnaseq \                                    # apptainer or singularity image
  -r 3.15.1 \                                         # release version
  -profile test \
  -c ./nextflow.config \                              # optional path to nextflow config (Default: $HOME/.config/nextflow/nextflow.config) 
  --outdir /scratch2/fs1/allocation_name/nextflow/output # root temp directory where nextflow work is performed
```

- In another terminal window you can verify that jobs are running in SLURM, and not locally, via `sacct`.

# Additional Information

- RIS does not support DSL1 based Nextflow pipelines.
