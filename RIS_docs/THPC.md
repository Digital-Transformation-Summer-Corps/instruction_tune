
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# THPC

- [What is the THPC?](#what-is-the-thpc)
- [Image Details](#image-details)
- [Environment Variables](#environment-variables)
- [Interactive Command-Line Session](#interactive-command-line-session)
- [Non-Interactive Batch Job](#non-interactive-batch-job)
- [Interactive GUI Session](#interactive-gui-session)
- [Useful Commands](#useful-commands)
- [Additional Software & Modules, Micro Architecture](#additional-software-modules-micro-architecture)
- [Available Versions](#available-versions)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# What is the THPC?

- THPC stands for traditional high performance computing. It is named as such as a reference to what can be considered a more traditional HPC environment.
- THPC uses the module system to establish and load software that users would like to utilize.
- It was developed for those more familiar with the module system and not familiar with Docker.
- The THPC image is different than other images, because it doesn’t install the software directly into the image.

  - The software is installed in a location that the THPC image can access.
  - The software is loaded into the image and job via the module system.
  - This is done to reduce the size of the image and reduce load times.

# Image Details

- Docker image hosted at ghcr.io/washu-it-ris/ris-thpc

# Environment Variables

These variables allow users to define the software release module path to be used in a job session or batch job.

> [!IMPORTANT]
> - Not every combination of variables is valid for a build release.
> - Only variable combinations aligning with the host micro architecture are compatible.

## THPC\_BUILD

RIS software build release formatted as *YYYY.MM*.

Current release: `2023.06`

View available build releases (can only be run inside a running job where /opt/thpc is mounted, i.e. THPC):

```
ls /opt/thpc/
```

## THPC\_MICRO

Micro architecture of host on which the job should run.

Examples: skylake, cascadelake, zen5

View all hosts in a queue with a given micro architecture (ex: skylake in general-interactive queue):

```
bhosts -R 'select[cpumicro==skylake]' general-interactive
```

Note that GPU hosts with a given micro architecture in the same queue (ex: skylake in general-interactive queue) may differ:

```
bhosts -R 'select[cpumicro==skylake]' general-interactive -gpu
```

## Default Values: GUI, Wrapper Scripts

- `THPC_BUILD`: latest released RIS build

# Interactive Command-Line Session

## Wrapper Script

`thpc-terminal` script starts a command-line session with default resource values.

```
thpc-terminal /bin/bash
```

## Basic Job Command

- Insert value for `THPC_BUILD`.
- Define cpumicro (same as `THPC_MICRO`) only if a specific micro architecture is required, otherwise can remove `-R "select[cpumicro==]"` entirely.

```
THPC_BUILD= \
THPC_CONTAINER_OS=linux \
LSF_DOCKER_VOLUMES=/opt/thpc:/opt/thpc bsub -Is -R "select[cpumicro==]" \
-q general-interactive -a 'docker(ghcr.io/washu-it-ris/ris-thpc:runtime)' /bin/bash
```

# Non-Interactive Batch Job

## thpc-batch Wrapper Script

`thpc-batch` script starts a batch job with default resource values. Replace `PROGRAM` with the program or script to be run.

```
thpc-batch PROGRAM
```

## Basic Job Command

- Insert value for `THPC_BUILD`.
- Define cpumicro (same as `THPC_MICRO`) only if a specific micro architecture is required, otherwise can remove `-R "select[cpumicro==]"` entirely.
- Replace `PROGRAM` with the program or script to be run.

```
THPC_BUILD= \
THPC_CONTAINER_OS=linux \
LSF_DOCKER_VOLUMES=/opt/thpc:/opt/thpc bsub -R "select[cpumicro==]" \
-q general -a 'docker(ghcr.io/washu-it-ris/ris-thpc:runtime)' PROGRAM
```

# Interactive GUI Session

- Connect to [http://ood.ris.wustl.edu](http://ood.ris.wustl.edu/).
- Select Compute RIS Desktop from the Interactive Apps dropdown.
- Enter resource requirements and storage/scratch directory mounts.
- Select Launch to submit job.

Additional information: [Open OnDemand](../../../Compute1/Open%20OnDemand.md).

# Useful Commands

THPC uses Lmod to dynamically control the environment through use of module files. See the [official docs](https://lmod.readthedocs.io/en/latest/010_user.html) for guidance beyond these basic commands.

Shorthand exists for these module commands as well. Enter ml -h in a job session or see the [official docs](https://lmod.readthedocs.io/en/latest/010_user.html#ml-a-convenient-tool) for more information.

## Available Modules

Loading modules without specifying a version will result in the default module being loaded. Default module versions are denoted with (D) in the listing if more than one version is available.

> ```
> module avail
> ```

## Loaded Modules

> ```
> module list
> ```

## Load/Unload Modules

> ```
> module load package1 package2 ...
> module unload package1 package2 ...
> ```

# Additional Software & Modules, Micro Architecture

## Open OnDemand Engineering Modules

- The THPC platform now has access to a build that has applications from the Engineering Group.
- These can be loaded by clicking the checkbox in job form in OOD.

![image-20250313-195407.png](../../../../attachments/d4d879dd-3057-4eec-a3fc-19f381ad95a3.png)

- Selecting this option does two things.

  - The Docker image becomes RHEL based instead of Ubuntu.
  - Allows access to community modules provided by the Engineering Group.
- If a user cannot run a module or encounters an error with a module, contact information for that module is available with the following command.

```
module help MODULENAME
```

- Where MODULENAME is replaced with the name of the module.
- Example:

```
$ module help keysight

------------------------------------ Module Specific Help for "keysight/2023" ------------------------------------
This module is a THPC community-managed licensed application. Contact Mark Bober at bober@wustl.edu for access.
```

## Extend Current/Previous Build Release

Current and previous build releases will not be globally extended by RIS.

Users can, however, install software and modules to their storageN environment to be used in conjunction with a current or previous build release.

Common options for installing software: - Python and [Create Custom Conda Environment](../../../Compute1/Create%20Custom%20Conda%20Environment.md). - [EasyBuild](https://docs.easybuild.io/) build and installation framework.

## Include Software, Modules in Upcoming Build Release

- To request RIS include additional software or modules in an upcoming build release, submit a request with details to the RIS Service Desk using [this form.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43).
- RIS intends to open the GitHub code repository to the community, allowing direct requests to add user-owned modules to the official RIS THPC build offerings. While not currently available, a communication will be sent to the RIS compute user community with instruction.

## New Build Release for Micro Architecture

RIS does not provide build releases compatible with all micro architectures by default. If the current build release is not available on a necessary micro architecture, submit a request with details to the RIS Service Desk using [this form.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43).

# Available Versions

## Current Version

- ghcr.io/washu-it-ris/ris-thpc

  - runtime
