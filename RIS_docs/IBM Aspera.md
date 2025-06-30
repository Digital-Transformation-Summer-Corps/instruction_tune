
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# IBM Aspera

- [Image Details](#image-details)
- [Interactive Job](#interactive-job)
- [Available Versions](#available-versions)

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

# Image Details

- Docker image hosted at <http://ghcr.io/washu-it-ris/aspera-connect>
- IBM Aspera Connect

# Interactive Job

## Environment Variables

- The Aspera SCP Password provided needs to be set as a variable.

```
export ASPERA_SCP_PASS="Replace With Provided Password"
```

- The directory where the provided ssh key is saved needs to be included.

  - E.g. The ssh key is saved in `/storageN/fs1/${ALLOCATION_NAME}/Active/aspera_key`

  ```
  export LSF_DOCKER_VOLUMES="/storageN/fs1/${ALLOCATION_NAME}/Active:/storageN/fs1/${ALLOCATION_NAME}/Active"
  ```
- A job should be submitted with the following information to create an interactive job.

```
bsub -q general-interactive -Is -a 'docker(ghcr.io/washu-it-ris/aspera-connect:ubuntu20)' /bin/bash
```

- Once the job is running, Aspera commands can be used. Like in the following.

```
ascp -i "/path/to/aspera_key/aspera_tokenauth_id_rsa" -Q -l 200m -k 1 test.file 'asp-dbgap@gap-submit.ncbi.nlm.nih.gov:test'
```

Download Files

- The example is for a test file to make sure Aspera works. Replace `'asp-dbgap@gap-submit.ncbi.nlm.nih.gov:test'` with the file to be downloaded.
- It is recommended that files are transfered individually and to use scripting loops to download batches.

## Docker Image

```
ghcr.io/washu-it-ris/aspera-connect:<tag>
```

# Available Versions

## Current Version:

- ghcr.io/washu-it-ris/aspera-connect

  - ubuntu20
