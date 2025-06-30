
[Compute Workshops](../Compute%20Workshops.md)

# Nvidia Workshop - September 2020

- [Quick Start](#quick-start)

> [!IMPORTANT]
> The compute group `compute-workshop` and the queues `workshop` and `workshop-interactive` are only available to those who partake in the workshop and only for a limited time. If you wish to use compute services beyond the workshops you will need to [sign up for access.](https://washu.atlassian.net/servicedesk/customer/portal/2)

# Quick Start

## Run on LSF

```
export JPORT=<port_number>
LSF_DOCKER_VOLUMES="$HOME:$HOME" LSF_DOCKER_PORTS="$JPORT:$JPORT" bsub -G compute-workshop -M 32GB -Is -R "gpuhost rusage[mem=32GB] select[port$JPORT=1]" -q workshop-interactive -gpu "num=1:gmodel=TeslaV100_SXM2_32GB" -a 'docker(gcr.io/ris-registry-shared/nvidia-workshop-sept-2020)' /opt/conda/bin/entrypoint.sh
```

- Please see our documentation for more information on selecting a port.
- For workshop purposes, the script was placed into a scratch1 space that should be accessible to all compute users:

```
export JPORT=<port_number>
/scratch1/fs1/ris/application/nvidia-workshop/start.sh
```

- Point browser to the URL given that starts `http://compute1-exec-<host>.ris.wustl.edu:...` where `<host>` is replaced by the exec node the job landed on. This is found in the terminal after the job command as `<<Starting on compute1-exec-<host>.ris.wustl.edu>>.`

## Run on LSF if not using the MedSchool VPN

- You can also connect to the gui with port forwarding on your local machine. :ref:`You can find the documentation on port forwarding [Port Forwarding](../Compute1/Port%20Forwarding.md)
- There are some slight differences when using the workshop queues. These are noted below.

  - Use the following ssh command instead of the one in the port forwarding documentation.

```
ssh -L $JPORT:compute1-exec-<host>.compute.ris.wustl.edu:$JPORT <wustlkey>@compute1-client-<new-host>.ris.wustl.edu
```

- Replace <hosts=> with the exec node where the job landed. This is found in the terminal after the job command as `<<Starting on compute1-exec-<host>.ris.wustl.edu>>`.
- Replace `<new-host>` with a number in the range 204-212.

```
Replace ``<host>`` with the exec node where the job landed. This is found in the terminal after the job command as ``<<Starting on compute1-exec-<host>.ris.wustl.edu>>``.
    - Replace ``<new-host>`` with a number in the range 204-212.
```

## Seminar Repository

The jupyter notebooks, code, and example data can be found [here.](https://bitbucket.ris.wustl.edu/projects/DOCK/repos/nvidia-workshop-sept-2020/browse)
