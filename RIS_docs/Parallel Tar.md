
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Parallel Tar

- [Overview](#overview)
- [Interactive Command-Line Session](#interactive-command-line-session)
- [Non-Interactive Session](#non-interactive-session)
- [Expected Output](#expected-output)
- [Script Variables](#script-variables)

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

# Overview

The purpose of this document is to demonstrate steps required to parallel tar using RIS-hosted image `gcr.io/ris-registry-shared/parallel-tar`. Input variable and process execution validation included.

# Interactive Command-Line Session

```
bsub -n 2 -Is -q general-interactive -a 'docker(gcr.io/ris-registry-shared/parallel-tar)' /bin/bash
```

```
parallel-tar.py -s path/to/source_dir/ -d path/to/dest_dir/
```

# Non-Interactive Session

```
bsub -n 2 -q general -a 'docker(gcr.io/ris-registry-shared/parallel-tar)' "parallel-tar.py -s path/to/source_dir/ -d path/to/dest_dir/"
```

# Expected Output

```
Running tar with parameters
source directory: path/to/source_dir/
destination directory: path/to/dest_dir/
threads: 4
tar file name: path/to/dest_dir/source_dir.tar.gz
remove source directory: False

Validating tar contents

Parallel-tar completed successfully
```

# Script Variables

- -s –source-dir source directory to tar and compress
- -d –dest-dir destination directory to place tar compressed file
- -t –threads number of threads for the compress algorithm, default 4
- -n –tar-file-name name of tar file with out the .tar.gz extension
- -r –remove-source-dir Remove the source directory when tar process completes, default False
