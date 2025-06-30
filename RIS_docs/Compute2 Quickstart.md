
[Compute2](../Compute2.md)

# Compute2 Quickstart

> [!WARNING]
> This is the place for documentation in regards to using the Compute2 Platform, part of RIS services and the future location of all RIS User Documentation. These documents are actively being developed and in flux.

- [Requirements:](#requirements)
- [Connect via SSH](#connect-via-ssh)
- [Using Slurm](#using-slurm)
  - [Load RIS module via ml](#load-ris-module-via-ml)
  - [Job Options](#job-options)
  - [Using srun](#using-srun)
  - [Using sbatch](#using-sbatch)
  - [Using squeue](#using-squeue)
  - [Using scancel](#using-scancel)
  - [Using sinfo](#using-sinfo)
  - [Using GPUs](#using-gpus)
  - [Using Containers](#using-containers)
  - [Using Job Arrays](#using-job-arrays)

# Requirements:

- [VPN](https://it.wustl.edu/items/connect/) (For off campus access)
- Terminal Access

  - Windows: PowerShell or downloaded software like [PuTTY](https://putty.org/) or [MobaXterm](https://mobaxterm.mobatek.net/)
  - Mac: Terminal (Under Utilities)
  - Linux: Terminal

# Connect via SSH

- Connect via 1 of 3 login clients

```
ssh washukey@c2-login-00N.ris.wustl.edu
```

> Where N is 1, 2, or 3.

# Using Slurm

## Load RIS module via `ml`

- See what is available via `ml` with `avail` command

```
$ ml avail
----------------------- /cm/shared/apps/mellanox/hpcx-v2.18.1-gcc-mlnx_ofed-redhat9-cuda12-x86_64/modulefiles -----------------------
hpcx  hpcx-debug  hpcx-debug-ompi  hpcx-mt  hpcx-mt-ompi  hpcx-ompi  hpcx-prof  hpcx-prof-ompi  hpcx-stack  

------------------------------------------------------- /cm/local/modulefiles -------------------------------------------------------
boost/1.81.0             cm-bios-tools  dot              ipmitool/1.8.19  mariadb-libs  null      python39                
cluster-tools-dell/10.0  cmd            freeipmi/1.6.10  lua/5.4.6        module-git    openldap  shared                  
cluster-tools/10.0       cmjob          gcc/13.1.0       luajit           module-info   python3   slurm/compute2/23.02.5  

------------------------------------------------------ /cm/shared/modulefiles -------------------------------------------------------
blacs/openmpi/gcc/64/1.1patch03  intel/compiler-rt/2024.0.2  intel/ifort/2024.2.1               intel/oclfpga/latest        
blas/gcc/64/3.11.0               intel/compiler-rt/2024.2.1  intel/ifort/latest                 intel/tbb/2021.11           
bonnie++/2.00a                   intel/compiler-rt/latest    intel/inspector/2024.0             intel/tbb/2021.13           
cm-pmix3/3.1.7                   intel/compiler/2024.0.2     intel/inspector/latest             intel/tbb/latest            
cm-pmix4/4.1.3                   intel/compiler/2024.2.1     intel/intel_ipp_intel64/2021.12    intel/vtune/2024.2          
default-environment              intel/compiler/latest       intel/intel_ipp_intel64/latest     intel/vtune/latest          
fftw3/openmpi/gcc/64/3.3.10      intel/debugger/2024.0.0     intel/intel_ippcp_intel64/2021.12  iozone/3.494                
gdb/13.1                         intel/debugger/2024.2.1     intel/intel_ippcp_intel64/latest   lapack/gcc/64/3.11.0        
globalarrays/openmpi/gcc/64/5.8  intel/debugger/latest       intel/itac/2022.0                  mpich/ge/gcc/64/4.1.1       
hdf5/1.14.0                      intel/dnnl/3.5.0            intel/itac/latest                  mvapich2/gcc/64/2.3.7       
hdf5_18/1.8.21                   intel/dnnl/latest           intel/mkl/2024.2                   netcdf/gcc/64/gcc/64/4.9.2  
hwloc/1.11.13                    intel/dpct/2024.2.0         intel/mkl/latest                   netperf/2.7.0               
hwloc2/2.8.0                     intel/dpct/latest           intel/mpi/2021.11                  openblas/dynamic/(default)  
intel/advisor/2024.2             intel/dpl/2022.3            intel/mpi/2021.13                  openblas/dynamic/0.3.18     
intel/advisor/latest             intel/dpl/2022.6            intel/mpi/latest                   openmpi/gcc/64/4.1.5        
intel/ccl/2021.13.1              intel/dpl/latest            intel/oclfpga/2024.0.0             openmpi4/gcc/4.1.5          
intel/ccl/latest                 intel/ifort/2024.0.2        intel/oclfpga/2024.2.1             ucx/1.10.1                  

```

- Load RIS module

```
ml load ris
```

## Job Options

- The help option provides a full list of what's available
- The basic options are listed here

  - CPU: Use `--ntasks` for increase of CPUs used with `--cpus-per-task=1`

  ```
  -c, --cpus-per-task=ncpus   number of cpus required per task
  -n, --ntasks=ntasks         number of tasks to run
  ```

  - Job name

  ```
  -J, --job-name=jobname      name of job
  ```

  - Ram/Memory

  ```
  --mem=MB                minimum amount of real memory
  --mem-per-cpu=MB        maximum amount of real memory per allocated
                          cpu required by the job.
                          --mem >= --mem-per-cpu if --mem is specified.
  ```

  - Host/Node/Server

  ```
  -N, --nodes=N               number of nodes on which to run (N = min[-max])
  ```

  - Partition/Queue (Default: general)

  ```
  -p, --partition=partition   partition requested
  ```

  - Standard Out (`stdout`)

  ```
  -o, --output=out            location of stdout redirection
  ```

  - Job Run Time

  ```
  -t, --time=minutes          time limit
  ```

  - Using a container (Docker image)

  ```
  --container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH
                          [pyxis] the image to use for the container
                          filesystem. Can be either a docker image given as
                          an enroot URI, or a path to a squashfs file on the
                          remote host filesystem.
  ```

  - Using a GPU

  ```
  -G, --gpus=n                count of GPUs required for the job
      --gpus-per-node=n       number of GPUs required per allocated node
      --mem-per-gpu=n         real memory required per allocated GPU
      --cpus-per-gpu=n        number of CPUs required per allocated GPU
  ```

## Using `srun`

- Submits a job that runs in real time
- Example using python

```
$ srun python helloworld.py 
Hello World!
```

- Storage allocations are mounted and you do not need to mount storage

  - Storage allocations do need to be mounted if using a container. See the [Using Containers](https://washu.atlassian.net/wiki/spaces/RUD/pages/edit-v2/1720615105#Using-Containers) section.
- You can get a shell connection to a job by using the `--pty` option and `/bin/bash` as the command.

```
srun --pty /bin/bash
```

- This works for jobs that use installed applications or containers.

## Using `sbatch`

- Submits a batch job (Runs in the background)
- Create a job file: testjob.slurm

```
#!/bin/bash
#SBATCH --job-name=python-test
#SBATCH --output=test.log

python helloworld.py
```

- Submit job using `sbatch`

```
$ sbatch testjob.slurm
```

- Check the output

```
$ cat test.log 
Hello World!
```

- Storage allocations are mounted and you do not need to mount storage.

  - Storage allocations do need to be mounted if using a container. See the [Using Containers](https://washu.atlassian.net/wiki/spaces/RUD/pages/edit-v2/1720615105#Using-Containers) section.

## Using `squeue`

- Displays running jobs
- Basic options listed here

  - By user

  ```
  -u, --user=user_name(s)         comma separated list of users to view
  ```

  - By partition/queue

  ```
  -p, --partition=partition(s)    comma separated list of partitions
  			         to view, default is all partitions
  ```

  - By name

  ```
  -n, --name=job_name(s)          comma separated list of job names to view
  ```

  - By account

  ```
  -A, --account=account(s)        comma separated list of accounts
  			         to view, default is all accounts
  ```
- By default shows all jobs

```
$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
               103   general python-t     elyn  R       0:11      1 c2-node-001
               104   general python-t     elyn  R       0:06      1 c2-node-001
               105   general osu_bibw   sleong  R       0:06      2 c2-node-[001,079]
               106   general python-t     elyn  R       0:03      1 c2-node-001
```

- Use the `-u` option to show only your jobs

```
$ squeue -u elyn
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
               103   general python-t     elyn  R       1:42      1 c2-node-001
               104   general python-t     elyn  R       1:37      1 c2-node-001
               106   general python-t     elyn  R       1:34      1 c2-node-001
```

## Using `scancel`

- Cancels or kills running jobs
- Default use job ID to cancel a job

```
scancel job_id[_array_id]
```

- Use the `--me` option to kill all your jobs

```
scancel --me
```

- Can cancel via other options

  - Job name

  ```
  -n, --name=job_name             act only on jobs with this name
  ```

  - User

  ```
  -u, --user=user_name            act only on jobs of this user
  ```

  - Partion/Queue

  ```
  -p, --partition=partition       act only on jobs in this partition
  ```

## Using `sinfo`

- Provides partitions/queues available

```
$ sinfo
PARTITION     AVAIL  TIMELIMIT  NODES  STATE NODELIST
bigmem           up   infinite     82   idle c2-bigmem-[001-002],c2-node-[001-080]
general*         up   infinite     80   idle c2-node-[001-080]
general-short    up      30:00     80   idle c2-node-[001-080]
gpu              up   infinite      2  down* c2-gpu-[010,012]
gpu              up   infinite     94   idle c2-gpu-[001-009,011,013-016],c2-node-[001-080]
```

## Using GPUs

- GPUs are available in the general and general-short partitions
- Use a script that uses pytorch to test for GPU usage called `test_gpu.py`

```
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- Load `slurm` and `pytorch` module. Shown below is an example command.

```
ml load slurm pytorch
```

- Run a test using a python script

```
srun -p general --gpus=1 python test_gpu.py
```

## Using Containers

- A container (Docker image) can be used with the following option

```
--container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH
                            [pyxis] the image to use for the container
                            filesystem. Can be either a docker image given as
                            an enroot URI, or a path to a squashfs file on the
                            remote host filesystem.
```

- `${Home}` directories are automatically mounted
- The `${Home}` directory is not default working directory
- Use the full path name for files

```
$ srun --container-image=python:3.9.21-alpine python /home/elyn/helloworld.py 
pyxis: importing docker image: python:3.9.21-alpine
pyxis: imported docker image: python:3.9.21-alpine
Hello World!
```

- The working directory can be set with the following option

```
--container-workdir=PATH
                        [pyxis] working directory inside the container
```

- Example setting the working directory

```
$ srun --container-image=python:3.9.21-alpine --container-workdir=/home/elyn python helloworld.py 
pyxis: importing docker image: python:3.9.21-alpine
pyxis: imported docker image: python:3.9.21-alpine
Hello World!
```

- Storage allocations are not mounted by default
- Storage allocations can be mounted with the following option.

```
--container-mounts=SRC:DST[:FLAGS][,SRC:DST...]
                        [pyxis] bind mount[s] inside the container. Mount
                        flags are separated with "+", e.g. "ro+rprivate"
```

- Example mounting a storage allocaiton

```
$ srun --container-image=python:3.9.21-alpine --container-mounts=/storage2/fs1/elyn/Active:/storage2/fs1/elyn/Active ls /storage2/fs1/elyn/Active
pyxis: importing docker image: python:3.9.21-alpine
pyxis: imported docker image: python:3.9.21-alpine
testfile.txt
```

- These same options can be used with `sbatch` submissions

## Using Job Arrays

- Job arrays can only be used with `sbatch` jobs
- They can be used with the following option

```
-a, --array=indexes         job array index values
```

- An example of using a job arrays

  - Job script

  ```
  #!/bin/bash
  #SBATCH --job-name=array-test
  #SBATCH --array=1-5
  #SBATCH --output=testarray.log

  /bin/bash -c "date;sleep 300;date"
  ```

  - Job submission

  ```
  $ sbatch testarrayjob.slurm 
  Submitted batch job 243
  ```

  - Looking at job arrays in the queue

  ```
  $ sbatch testarrayjob.slurm 
  Submitted batch job 243
  [elyn@c2-login-001 ~]$ squeue -u elyn
           JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           243_2   general array-te     elyn  R       0:09      1 c2-node-001
           243_3   general array-te     elyn  R       0:09      1 c2-node-001
           243_4   general array-te     elyn  R       0:09      1 c2-node-001
           243_5   general array-te     elyn  R       0:09      1 c2-node-001
           243_1   general array-te     elyn  R       0:10      1 c2-node-001
  ```
- Job arrays can all be cancelled

  - Example for job array (job ID) 50 with 5 elements

  ```
  scancel 50
  ```
- Or particular array elements can be cancelled

  ```
  scancel 50_1

  scancel 50_[1-3]

  scancel 50_1 50_4
  ```
