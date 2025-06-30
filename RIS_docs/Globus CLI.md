
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Globus CLI

- [Software Included](#software-included)
- [Development Notes](#development-notes)
- [Dockerfile](#dockerfile)
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

# Software Included

- vim (<https://github.com/vim/vim> )
- Globus CLI (<https://docs.globus.org/cli/installation/>)
- virtualenv (<https://virtualenv.pypa.io/en/latest/> )

# Development Notes

- The Globus CLI is incompatible with Python 2.7 (Python 2.7 should no longer be used in general), as thus uses Python3.

# Dockerfile

```
# Set the base image to Ubuntu:xenial
FROM ubuntu:xenial


# Update the repository sources list
# Install base packages: pip, git, vim
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    git \
    vim \
    libffi-dev

#Obtain and run get-pip.py necessary for Globus CLI install
WORKDIR /opt/
#ADD https://bootstrap.pypa.io/get-pip.py .
ADD https://bootstrap.pypa.io/pip/3.5/get-pip.py .
RUN python3 get-pip.py

#Create environment settings necessary for Globus CLI install
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV GLOBUS_CLI_INSTALL_DIR="$(python -c 'import site; print(site.USER_BASE)')/bin"
RUN echo "GLOBUS_CLI_INSTALL_DIR=$GLOBUS_CLI_INSTALL_DIR"
ENV PATH="$GLOBUS_CLI_INSTALL_DIR:$PATH"
RUN echo 'export PATH="'"$GLOBUS_CLI_INSTALL_DIR"':$PATH"' >> "$HOME/.bashrc"

#Install Globus CLI
RUN pip install --upgrade globus-cli

#Install virtualenv
RUN pip install virtualenv

#Pull in necessary automation examples
RUN git clone https://github.com/globus/automation-examples
WORKDIR automation-examples
RUN chmod +x cleanup_cache.py cli-sync.sh globus_folder_sync.py share-data.sh share_data.py

#Run and install requirements for virtualenv
RUN virtualenv venv
RUN . venv/bin/activate
RUN pip install -r requirements.txt
```

# Requirements for Usage in RIS

- The Globus CLI needs particular environment variables to be able to function and so you need to make sure you use the LSF variable to turn off the override of environment variables.

```
export LSF_DOCKER_PRESERVE_ENVIRONMENT=false
bsub -Is -G group-name -q general-interactive -a 'docker(location-of-docker-image)' /bin/bash
```

- Once the job is running you’ll need to start up Globus with the login command.

```
globus login --no-local-server
```

- This will generate commands for you to follow to go to the web portal for Globus to connect your account with the command line.
- In all future instances of using the Globus CLI you’ll need to be in the directory you started the Globus CLI job in the first time for it to function properly. If not you’ll get an error.

```
You are running 'globus login', which should automatically open a browser window for you to login.
If this fails or you experience difficulty, try 'globus login --no-local-server'
---
PermissionError: [Errno 13] Permission denied: '/home/elyn/.globus.cfg'
```

- Once the transfers are submitted to Globus via the CLI commands, you do not need to keep the interactive job running.
- For more information on using the Globus CLI [Moving Data With Globus CLI](../../../Storage%20Platforms/Moving%20Data%20With%20Globus%20CLI.md)
