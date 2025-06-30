
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# AFNI/TORTOISE

- [Image Details](#image-details)
- [Requirements for Usage in RIS](#requirements-for-usage-in-ris)

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

# Image Details

- Docker image hosted at `gcr.io/ris-registry-shared/afni-tortoise` .
- Versions Available:

  - latest, v3.2.0
  - v3.1.4
- Software Included

  - AFNI
  - TORTOISE, comprised of the following tools:

    - DIFFPREP
    - DRBUDDI
    - DIFFCALC - CLI version
    - DRTAMAS
  - R Packages
  - GUI is not currently included in this image

# Requirements for Usage in RIS

## Initial Setup

- An initial setup of dependent files and directory structure in your compute home directory is required.

> [!IMPORTANT]
> - Setup will create/modify the following in the compute home directory: `.afni/`, `.afnirc`, `.bashrc`, `.sumarc`.
> - In rare cases, $HOME is not /home/washukey. See the [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md) for details. If this is the case, enter echo $HOME and insert the result at the appropriate places below.

- Connect to compute and run the following, adjusting for individual compute username, home location, etc.

  ```
  LSF_DOCKER_VOLUMES='/home/washukey:/home/washukey' PATH=/home/washukey:/linux_centos_7_64:$PATH
  bsub -q general-interactive -Is -a 'docker(gcr.io/ris-registry-shared/afni-tortoise)' /bin/bash -c "cp /linux_centos_7_64/AFNI.afnirc ~/.afnirc; suma -update_env; apsearch -update_all_afni_help; /bin/bash"
  ```

> [!IMPORTANT]
> - If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.
> - Paste the following in its entirety and run to append the .bashrc file.

```
cat <<EOF >> $HOME/.bashrc
ahdir=`apsearch -afni_help_dir`
if [ -f "$ahdir/all_progs.COMP.bash" ]
then
    . $ahdir/all_progs.COMP.bash
fi
export R_LIBS=/usr/local/lib/R/library/’
EOF
```

- Update shell.

```
. ~/.bashrc
```

## Run on LSF

- Connect to compute and run the following, adjusting for individual parameters and replacing <tag> with the appropriate version.

```
LSF_DOCKER_VOLUMES='/home/washukey:/home/washukey' PATH=/home/washukey:/linux_centos_7_64:$PATH
bsub -q general-interactive -Is -a 'docker(gcr.io/ris-registry-shared/afni-tortoise:<tag>)' /bin/bash
```
