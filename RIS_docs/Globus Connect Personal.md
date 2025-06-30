
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Globus Connect Personal

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

- tk
- tcllib
- wget
- python3
- Globus Connect Personal (<https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz>)

# Development Notes

- This image is designed for transfer of data to local or scratch storage only.

# Dockerfile

- The Globus Connect Personal Docker image is hosted at the following location.

```
gcr.io/ris-registry-shared/globus
```

# Requirements for Usage in RIS

## Setting up on Compute1

- Submit a job to run container with the following command.
- Add any paths you want to access from the Globus client (local or scratch) and your HOME directory is required.

```
PATH=/opt/globus/globusconnectpersonal-3.1.4:$PATH LSF_DOCKER_VOLUMES="$HOME:$HOME /path/to/directory:/path/to/directory" bsub -q general-interactive -Is -a 'docker(gcr.io/ris-registry-shared/globus)' /bin/bash
```

- `/path/to/directory` in the command is the path of the local or scratch directory that you wish to enable as an endpoint in Globus.

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.

- While inside the container, setup can be started with the following command.

```
globusconnectpersonal -setup
```

- This will start a guide that goes through the steps necessary for setting up Globus Connect Personal and will look like the following.

```
Globus Connect Personal needs you to log in to continue the setup process.
We will display a login URL. Copy it into any browser and log in to get a
single-use code. Return to this command with the code to continue setup.
Login here:
-----
https://auth.globus.org/v2/oauth2/authorize?client_id=4d6448ae-8ca0-40e4-aaa9-
8ec8e8320621&redirect_uri=https%3A%2F%2Fauth.globus.org%2Fv2%2Fweb%2Fauth-code&scope=openid+profile+urn%
3Aglobus%3Aauth%3Ascope%3Aauth.globus.org%3Aview_identity_set+urn%3Aglobus%3Aauth%3Ascope%3Atransfer.api.
globus.org%
3Agcp_install&state=_default&response_type=code&code_challenge=Cqt4heM3uz7n_QRRbNjtXheBckZVf9UESZREwjc4xVA&co
de_challenge_method=S256&access_type=online&prefill_named_grant=compute1-exec-N.ris.wustl.edu
-----
Enter the auth code: hIdkJ4jptWByPKvlbKUNvNFzcy4DiL
== starting endpoint setup
Input a value for the Endpoint Name: username-personal-endpoint-compute1-exec-N
registered new endpoint, id: ace0c3fc-a3b7-11eb-92d2-6b08dd67ff48
setup completed successfully
```

- The `compute1-exec-N` that is used comes from the node the job lands on.
- This is important to remember for the next step and any future use of Globus through this method.

## Running on Compute1

- After the initial setup, the following command can be used to start up the client at any time.

```
PATH=/opt/globus/globusconnectpersonal-3.1.4:$PATH LSF_DOCKER_VOLUMES="$HOME:$HOME /path/to/directory:/path/to/directory" bsub -q general -m compute1-exec-N -a 'docker(gcr.io/ris-registry-shared/globus)' globusconnectpersonal -start -restrict-paths rw~/,rw/path/to/directory,/tmp
```

- `/path/to/directory` is the same directory and path that was enabled in the setup part of the process.
- `compute1-exec-N` needs to be replaced with the same node name that the setup step ran on.
- N represents a number that is the number of one of the nodes.

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.

- The endpoint that is the local or scratch directory can be now accessed via the web app and used for data transfer.
- <https://app.globus.org/>
