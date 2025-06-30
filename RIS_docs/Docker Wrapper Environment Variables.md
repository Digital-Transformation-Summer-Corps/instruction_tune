
[Compute1](../Compute1.md)

# Docker Wrapper Environment Variables

- [LSF\_DOCKER\_ADD\_HOST](#lsf-docker-add-host)
- [LSF\_DOCKER\_ADD\_HOST\_FILE](#lsf-docker-add-host-file)
- [LSF\_DOCKER\_CGROUP](#lsf-docker-cgroup)
- [LSF\_DOCKER\_ENTRYPOINT](#lsf-docker-entrypoint)
- [LSF\_DOCKER\_ENV\_FILE](#lsf-docker-env-file)
- [LSF\_DOCKER\_IPC](#lsf-docker-ipc)
- [LSF\_DOCKER\_NETWORK](#lsf-docker-network)
- [LSF\_DOCKER\_PORTS](#lsf-docker-ports)
- [LSF\_DOCKER\_PRESERVE\_ENVIRONMENT](#lsf-docker-preserve-environment)
- [LSF\_DOCKER\_RUN\_LOGLEVEL](#lsf-docker-run-loglevel)
- [LSF\_DOCKER\_SHM\_SIZE](#lsf-docker-shm-size)
- [LSF\_DOCKER\_WORKDIR](#lsf-docker-workdir)
- [LSF\_DOCKER\_VOLUMES](#lsf-docker-volumes)
- [LSF\_DOCKER\_RUN\_RAW\_ARGS](#lsf-docker-run-raw-args)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

Features of the `docker` application (a.k.a. the docker wrapper) are accessed by setting environment variables before executing `bsub`. All the examples below show how to set the values inline with submitting the job, and multiples of these settings are given by including all of them on the same command-line:

```
LSF_DOCKER_OPTTION_1=value LSF_DOCKER_OPTION_2="value with spaces" bsub ...
```

Or they can also be set within the shell by “exporting” them, essentially making them apply to every command that follows:

```
[user@compute1-client-1 ~]$ export LSF_DOCKER_OPTION_1=value
[user@compute1-client-1 ~]$ export LSF_DOCKER_OPTION_2="value with spaces"
[user@compute1-client-1 ~]$ bsub ...
```

# LSF\_DOCKER\_ADD\_HOST

Adds one more more host -> IPaddress mappings to the docker container. The value of `LSF_DOCKER_ADD_HOST` is a space-separated list of mappings in the same format as the `--add-host` argument of `docker run`, ie. `hostname:IPaddress`

```
LSF_DOCKER_ADD_HOST="host_a:192.168.1.2 host_b:172.16.10.2" bsub ...
```

# LSF\_DOCKER\_ADD\_HOST\_FILE

Specify one more more space-separated files containing host -> IPaddress mappings. These files should have one mapping per line in the same format used by `LSF_DOCKER_ADD_HOST`:

```
host_a:192.168.1.2
host_b:172.16.10.2
```

These file names are then included in the `LSF_DOCKER_ADD_HOST_FILE` variable:

```
LSF_DOCKER_ADD_HOST_FILE="file_a file_b" bsub ...
```

# LSF\_DOCKER\_CGROUP

Set the value for the `--cgroup-parent` argument to `docker run`. There is no mechanism for creating custom cgroups at this time, so this should not be used unless specifically directed to by RIS.

# LSF\_DOCKER\_ENTRYPOINT

Overrides the container’s default `ENTRYPOINT`, and becomes the `--entrypoint` argument to `docker run`

```
LSF_DOCKER_ENTRYPOINT=/bin/bash bsub ...
```

# LSF\_DOCKER\_ENV\_FILE

Used to set envrionment variables within the running container. Expects a space-separated list of file names containing environment variables. The files should have one variable per line, and be formatted as `variable=value`:

```
SOME_VARIABLE=value
ANOTHER_VARIABLE=value with spaces
```

To make use of these files, set the `LSF_DOCKER_ENV_FILE` and submit the job:

```
LSF_DOCKER_ENV_FILE="env_file1 env_file2" bsub -q general -a 'docker(...)' ...
```

The file names are then used as `--env-file` arguments to `docker run`.

# LSF\_DOCKER\_IPC

Control the `--ipc` argument to `docker run`. This will become useful when building *MPI* applications.

```
export LSF_DOCKER_IPC=host
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

# LSF\_DOCKER\_NETWORK

Make use of docker’s `--network` argument. Most notablly for `--network=host`. See the [Docker networking tutorial](https://docs.docker.com/network/network-tutorial-host/).

```
export LSF_DOCKER_NETWORK=host
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

# LSF\_DOCKER\_PORTS

Specify network ports to expose from your docker container. Note that these must come from a range of “approved ports” (8000-8999) and should be “reserved” by the job scheduler. Here we map port 80 inside the container to port 8001 on the execution node, and we “reserve” port 8001 on that execution node through the scheduler.

```
export LSF_DOCKER_PORTS='8001:80'
bsub -q general -R 'select[port8001=1]' -a 'docker(httpd)' /usr/local/bin/httpd-foreground
```

The `LSF_DOCKER_PORTS` variable works with most expressions accepted by the `-p` option of `docker run`:

- `8001`: Expose a single port
- `8001:80`: Expose port 8001 and forward it to port 80 within the container
- `8001-8010`: Expose a range of ports
- `8001-8010:2001-2010`: Expose a range of forwarded ports
- `8001/tcp`: Expose only TCP port 8001
- `8001-8010/udp`: Expose a range of UDP ports

Additionally, more than one port may be exposed by separating each request with a space. For example:

```
LSF_DOCKER_PORTS='8000 8500 8600-8700`
```

See also [Job Execution Examples](Job%20Execution%20Examples.md).

> [!IMPORTANT]
> Note: LSF\_DOCKER\_PORTS is affected by the LSF\_DOCKER\_NETWORK type, and does not work with “host” type networking.

# LSF\_DOCKER\_PRESERVE\_ENVIRONMENT

This is a boolean value instructing the job launcher to either preserve your shell environment or not. Valid values are the strings `true` and `false`. The default value is `true`.

When false, the container will have a minimal environment with only a few variables, including `HOSTNAME` set to the exec host the job runs on, and those variables defined in the container’s Dockerfile with `ENV`.

When true, the container will inherit most environment variables set when the job is submitted, except for a few:

- HOSTNAME - becomes the exec host’s hostname
- LSB\_INTERACTIVE - removed within an interactive job so jobs started within that job are not interactive by default
- values containing newlines - these variables are removed due to a bug in docker’s handling of environment variables

```
export LSF_DOCKER_PRESERVE_ENVIRONMENT=false
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

# LSF\_DOCKER\_RUN\_LOGLEVEL

Set the log level for the wrapper script:

```
export LSF_DOCKER_RUN_LOGLEVEL=DEBUG
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

Valid values are, in descending order of verbosity: CRITICAL, ERROR, WARNING, INFO, DEBUG

# LSF\_DOCKER\_SHM\_SIZE

Control the `--shm-size` argument to `docker run`. This will become useful when building *MPI* applications.

```
export LSF_DOCKER_SHM_SIZE=4g
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

# LSF\_DOCKER\_WORKDIR

The default initial working directory within the container is the same as the working directory when the job was submitted. This can be overridden with `LSF_DOCKER_WORKDIR` and becomes the `-w` option to `docker run`, and must be a directory accessible within the container.

```
LSF_DOCKER_WORKDIR=/path/within/container bsub ...
```

# LSF\_DOCKER\_VOLUMES

This is a space separated list of filesystem locations to pass into your docker container by means of the `--mount` flag. The format is `src:dst`, where `src` means the filesystem location outside the container should be mapped to `dst` inside the container. (See also `--mount` at [Docker Volumes](https://docs.docker.com/storage/volumes/).)

```
export LSF_DOCKER_VOLUMES="/storageN/fs1/${STORAGE_ALLOCATION}:/storageN/fs1/${STORAGE_ALLOCATION} /scratch1/fs1/${COMPUTE_ALLOCATION}:/scratch1/fs1/${COMPUTE_ALLOCATION}"
bsub -Is -G ${group_name} -q general-interactive -a 'docker(ubuntu)' /bin/bash
```

> [!IMPORTANT]
> Elements in LSF\_DOCKER\_VOLUMES must be *directories*. Docker would allow you to pass in files, such that the following might be expected to work:
>
> LSF\_DOCKER\_VOLUMES=”$HOME/etc/myfile:/etc/somefile”
>
> But the RIS environment explicitly prevents this due to a security vulnerability.

A volume can be mounted read-only by appending `:ro` to any of the mounts, for example: `src:dst:ro`.

# LSF\_DOCKER\_RUN\_RAW\_ARGS

This variable tells bsub to use the command or entrypoint within the Docker image instead of requesting the command or entrypoint from the user as is the regular interaction without this variable.

This means that Docker images like the basic hello-world Docker image can be run on the Compute Platform.

The command also makes use of `LSB_DOCKER_PLACE_HOLDER` which is, as the name suggests, just a placeholder as bsub still needs an input to run. This is NOT passed as a command to the Docker container.

Example Command:

```
LSF_DOCKER_RUN_RAW_ARGS=1 bsub -Is -G ${group_name} -q general-interactive -a 'docker(hello-world)' LSB_DOCKER_PLACE_HOLDER
```

Output:

```
Job <897631> is submitted to queue <qa-interactive>.
<<Waiting for dispatch ...>>
<<Starting on compute1-exec-83.ris.wustl.edu>>
Using default tag: latest
latest: Pulling from library/hello-world
Digest: sha256:ffb13da98453e0f04d33a6eee5bb8e46ee50d08ebe17735fc0779d0349e889e9
Status: Image is up to date for hello-world:latest
docker.io/library/hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```
