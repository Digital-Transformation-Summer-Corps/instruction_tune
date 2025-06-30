
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# fsl6

- [Software Included](#software-included)
- [Dockerfile](#dockerfile)
- [Run on LSF](#run-on-lsf)

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

# Software Included

- Functional MRI: FEAT, MELODIC, FABBER, BASIL, VERBENA
- Structural MRI: BET, FAST, FIRST, FLIRT & FNIRT, FSLVBM, SIENA & SIENAX, MIST, BIANCA, MSM, fsl\_anat
- Diffusion MRI: FDT, TBSS, XTRACT, eddy, topup, eddyqc
- GLM /Stats: GLM general advice, Randomise, PALM, Cluster, FDR, Dual Regression, Mm, FLOBS
- Other: FSLeyes, FSLView, Fslutils, Atlases, Atlasquery, SUSAN, FUGUE, MCFLIRT, Miscvis, POSSUM, BayCEST

# Dockerfile

### Versions

- Docker image hosted at

  ```
  gcr.io/ris-registry-shared/fsl6
  ```
- Minimum Docker image hosted at

  ```
  gcr.io/ris-registry-shared/fsl6_min
  ```
- Code repository is located at [BitBucket](https://bitbucket.wustl.edu/projects/RISDEV/repos/bayly-fsl/browse) and requires WashU VPN connection and a washukey to access.

# Run on LSF

## Preserve Environment variables

- Paths inside the container are not being preserved unless one sets this variable for the whole session.

  - export LSF\_DOCKER\_PRESERVE\_ENVIRONMENT=false
  - Prepend LSF\_DOCKER\_PRESERVE\_ENVIRONMENT=false before an individual bsub command (in which case the variable is only false for that command).
- If one fails to set preserve environment variable to false, one would have to provide each application’s full path.

## bsub Command

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.

- Interactive Session

  ```
  bsub -Is -q general-interactive -a 'docker(gcr.io/ris-registry-shared/<VERSION>)' /bin/bash
  ```
- Alternatively, if you want the preserve environment to only be false for one command.

  ```
  LSF_DOCKER_PRESERVE_ENVIRONMENT=false bsub -Is -q general-interactive -a 'docker(gcr.io/ris-registry-shared/<VERSION>)' /bin/bash
  ```
- Non-Interactive

  ```
  bsub -a 'docker(gcr.io/ris-registry-shared/<VERSION>)' {command & arguments to run}
  ```
