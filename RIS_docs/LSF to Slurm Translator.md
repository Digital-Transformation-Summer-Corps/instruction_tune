
[Compute2](../Compute2.md)

# LSF to Slurm Translator

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Enable the lsf2slurm module.](#enable-the-lsf2slurm-module)
- [Run a job with bsub on the Compute2 cluster.](#run-a-job-with-bsub-on-the-compute2-cluster)
- [Check jobs with bjobs.](#check-jobs-with-bjobs)
- [Kill a job with bkill.](#kill-a-job-with-bkill)
- [Slurm Commands from LSF](#slurm-commands-from-lsf)

There is a simple LSF to Slurm translator that you can use to translate simple LSF `bsub`, `bjobs`, and `bkill` commands to Slurm commands. Shown below are step-by-step instructions to load the translator module and example commands to translate LSF commands to Slurm commands.

> [!IMPORTANT]
> Please keep in mind that this translator is not fully implemented all LSF commands and their options to translate to Slurm commands and options. Therefore, it might not work correctly as you expected.

> [!WARNING]
> This translator is a use at your own risk type of script. If you want to learn more about Slurm, please see other documentation:
>
> - [Transitioning Between slurm and LSF](Transitioning%20Between%20slurm%20and%20LSF.md)
> - [Compute2 Quickstart](Compute2%20Quickstart.md)

# Enable the `lsf2slurm` module.

```
ml load ris lsf2slurm
```

# Run a job with `bsub` on the Compute2 cluster.

Interactive Example:

```
[sleong@c2-login-001 ~]$ bsub -Is -a "docker(ubuntu)" bash -c "cat /etc/os-release; hostname"
pyxis: importing docker image: ubuntu
pyxis: imported docker image: ubuntu
PRETTY_NAME="Ubuntu 24.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.1 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
c2-node-003
```

Batch Example:

```
[sleong@c2-login-001 ~]$ bsub -a "docker(ubuntu)" sleep infinity 
Submitted batch job 15584
```

# Check jobs with `bjobs`.

```
[sleong@c2-login-001 ~]$ bjobs -u sleong
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             15568   general vscode_U   sleong  R    4:57:33      1 c2-node-003
             15584   general submit.s   sleong  R       0:56      1 c2-node-003
[sleong@c2-login-001 ~]$ bjobs -u sleong 15584
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             15584   general submit.s   sleong  R       1:29      1 c2-node-003
[sleong@c2-login-001 ~]$ bjobs -l -u sleong 15584
Wed Apr 02 14:34:26 2025
             JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
             15584   general submit.s   sleong  RUNNING       2:31 UNLIMITED      1 c2-node-003
```

# Kill a job with `bkill`.

> [!IMPORTANT]
> Please take great care of the `bkill` command since it is not fully implemented to translate LSF `bkill` to Slurm. herefore, for this command, please run with `--dryrun` always for information purposes.

```
[sleong@c2-login-001 ~]$ bkill -s 9 15584
[sleong@c2-login-001 ~]$ bjobs -w 15584
             JOBID PARTITION                                     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)

```

# Slurm Commands from LSF

The Slurm command to use can be obtained by adding the option `--dryrun` to the bsub command.

```
[sleong@c2-login-001 ~]$ bsub --dryrun -Is -a "docker(ubuntu)" bash -c "cat /etc/os-release; hostname"
srun --pty --container-image=ubuntu --container-workdir=/home/sleong --container-mounts=/home/sleong:/home/sleong /home/sleong/.jobtmpdir/637c7c35-821f-4e22-9acc-e4a1fcfe7afa/submit.sh
[sleong@c2-login-001 ~]$ bjobs --dryrun -u sleong 15584
squeue -u sleong --job 15584
[sleong@c2-login-001 ~]$ bkill --dryrun -s 9 15584
scancel -s 9 15584
```
