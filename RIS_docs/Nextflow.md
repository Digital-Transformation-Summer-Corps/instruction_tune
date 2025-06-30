
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Nextflow

- [Interactive GUI Session](#interactive-gui-session)
- [Command-Line Sessions](#command-line-sessions)
- [Example Config File](#example-config-file)

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

# Interactive GUI Session

- Interactive GUI sessions are done via Open On Demand (OOD).

  - You can use the `Compute RIS Desktop` application.
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md).
- Fill out the fields with the appropriate information (explained in the quick start).
- Launch the `Compute RIS Desktop` application, and follow the steps below.

  - Include the following in the `Mounts` field.

  ```
  /scratch1/fs1/ris:/scratch1/fs1/ris
  ```

  - Once in an interactive session the following is an example command of using Nextflow.

  ```
  nextflow run nf-core/rnaseq \
  -profile test \
  -c /scratch1/fs1/ris/application/nextflow/conf/rnaseq.config \
  --outdir /path/to/outdir
  ```

# Command-Line Sessions

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.

## Interactive Session

- If users wish to use Nextflow in an interactive command-line session, it can be done via the THPC terminal.
- Users can find out about using the THPC terminal [THPC](THPC.md)
- Users can use Nextflow just like with OOD.
- Users must include `/scratch1/fs1/ris:/scratch1/fs1/ris` in the `LSF_DOCKER_VOLUMES` environment variable.
- Make sure the directory to be used as the Nextflow working directory is included in the `LSF_DOCKER_VOLUMES` environment variable.

```
LSF_DOCKER_VOLUMES="/scratch1/fs1/ris:/scratch1/fs1/ris /storageN/fs1/${STORAGE_ALLOCATION}/Active:/storageN/fs1/{$STORAGE_ALLOCATION}/Active" \
thpc-terminal bash -c "cd /path/to/nextflow/working/directory ; nextflow run nf-core/sarek -profile test -c /scratch1/fs1/ris/application/nextflow/conf/rnaseq.config --outdir /path/to/nextflow/working/directory"
```

- `/path/to/nextflow/working/directory` is the path to the Nextflow working directory.

## Batch Session

- If users wish to use Nextflow in an batch session, it can be done via the THPC batch.
- Users can find out about using the THPC terminal [THPC](THPC.md)
- Users must include `/scratch1/fs1/ris:/scratch1/fs1/ris` in the `LSF_DOCKER_VOLUMES` environment variable.
- Make sure the directory to be used as the Nextflow working directory is included in the `LSF_DOCKER_VOLUMES` environment variable.

```
LSF_DOCKER_VOLUMES="/scratch1/fs1/ris:/scratch1/fs1/ris /storageN/fs1/${STORAGE_ALLOCATION}/Active:/storageN/fs1/{$STORAGE_ALLOCATION}/Active" \
thpc-batch bash -c "cd /path/to/nextflow/working/directory ; nextflow run nf-core/sarek -r 3.4.1 -profile test -c /scratch1/fs1/ris/application/nextflow/conf/rnaseq.config --outdir /path/to/nextflow/working/directory"
```

> [!IMPORTANT]
> - `/path/to/nextflow/working/directory` is the path to the Nextflow working directory.
> - When changing to the pipeline of choice, remove the `-r 3.4.1` option as this is specific to the example.

# Example Config File

- The example config file below shows how the config file needs to be set up to interact with the Compute Platform

```
process {
    executor = "lsf"
    queue    = { "general" }
    clusterOptions =  { "-a 'docker(quay.io/${task.container})' -G compute-ris -J rnaseq -env 'all,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'" }
}

executor {
    queueSize = 7
    submitRateLimit = '1/1sec'
}
```
