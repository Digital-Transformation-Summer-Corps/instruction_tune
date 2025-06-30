
[Compute2](../Compute2.md)

# Using Slurm in Containers

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Use Slurm Commands in Containers](#use-slurm-commands-in-containers)
- [Install Slurm in Containers (Recommended Best Practice)](#install-slurm-in-containers-recommended-best-practice)

# Use Slurm Commands in Containers

By default Slurm commands are not available inside container jobs. If you want to use Slurm commands in container jobs, you will need to either install Slurm on your container images (recommended) or mount the host Slurm installation to your containers (not recommended). Care must be taken if you decide to use Slurm installation on the bare metal/host in your container jobs. You must make sure your container environments can support the bare metal/host Slurm installation. Shown below are details procedures for each method.

# Install Slurm in Containers (Recommended Best Practice)

1. Check the version of Slurm installation. Shown below is an example command.

   ```
   [sleong@c2-login-001 ~]$ sinfo --version
   slurm 23.02.5
   ```
2. Install munge to your containers. Shown below is an example Ubuntu munge installation command.

   ```
   apt-get install libmunge2 libmunge-dev
   ```
3. Install the same version of Slurm to your containers.
4. Add `slurm` user to your `/etc/passwd` file in your containers. Shown below is an example `/etc/passwd` entry for `slurm` user.

   ```
   slurm:x:450:450::/cm/local/apps/slurm:/bin/bash
   ```
5. Start your container jobs. Shown below is an example command.

   ```
   srun --pty --container-mounts=/run/munge --container-env=SLURM_CONF --container-image=mycontainer-with-slurm /bin/bash
   ```
