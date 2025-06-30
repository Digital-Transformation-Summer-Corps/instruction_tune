
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# RoseTTAFold All-Atom

- [Image Details](#image-details)
- [Running RoseTTAFold All-Atom Non-interactively](#running-rosettafold-all-atom-non-interactively)
- [Available RoseTTAFold Versions](#available-rosettafold-versions)

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

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Image Details

- Includes RoseTTAFold All-Atom

## Docker Image

```
ghcr.io/washu-it-ris/rosettafold-all-atom-docker:<tag>
```

> [!IMPORTANT]
> Docker Tag
>
> The `<tag>` will refer to the version of the Docker container. Please see below for the available versions.

# Running RoseTTAFold All-Atom Non-interactively

- Copy the script runscript.sh

```
#! /bin/bash

cd /app/RoseTTAFold/RoseTTAFold-All-Atom/
source /app/RoseTTAFold/mambaforge/condabin/activate RFAA
python -m rf2aa.run_inference --config-name $1
```

- Copy the script bsub-non-interactive.sh

```
#! /bin/bash

# SPECIFY ACCORDINGLY
COMPUTE_GROUP=MY-COMPUTE-GROUP

# where are the fasta or sfd files among others?
INPUT_PATH=MY-INPUT-PATH

# where do you want the resulting files?
OUTPUT_PATH=MY-OUTPUT-PATH

# where are the config inference files?
INFERENCES_PATH=MY-CONFIG-INFERENCE-PATH

# Required by LSF
QUEUE=general
MEMORY=64GB
SLOTS=4

# The image tag should be a tag of the rosettafold-all-atom-docker
# or an image based on the rosettafold-all-atom-docker image.
IMAGE_TAG="ghcr.io/washu-it-ris/rosettafold-all-atom-docker:1.0.0"

# DO NOT CHANGE BELOW THIS LINE
ROSETTAFOLD_ASSETS_PATH="/scratch1/fs1/ris/references/RoseTTAFold/bfd:/app/RoseTTAFold/RoseTTAFold-All-Atom/bfd \
    /scratch1/fs1/ris/references/RoseTTAFold/checkpoint:/app/RoseTTAFold/RoseTTAFold-All-Atom/checkpoint \
    /scratch1/fs1/ris/references/RoseTTAFold/pdb100_2021Mar03:/app/RoseTTAFold/RoseTTAFold-All-Atom/pdb100_2021Mar03 \
    /scratch1/fs1/ris/references/RoseTTAFold/UniRef30_2020_06:/app/RoseTTAFold/RoseTTAFold-All-Atom/UniRef30_2020_06 \
    /scratch1/fs1/ris/references/RoseTTAFold/blast-2.2.26:/app/RoseTTAFold/RoseTTAFold-All-Atom/blast-2.2.26 \
    /scratch1/fs1/ris/references/RoseTTAFold/csblast-2.2.3:/app/RoseTTAFold/RoseTTAFold-All-Atom/csblast-2.2.3"

USER_CONFIGURED_PATH="$OUTPUT_PATH:/app/RoseTTAFold/RoseTTAFold-All-Atom/outputs \
    $INPUT_PATH:/app/RoseTTAFold/RoseTTAFold-All-Atom/inputs \
    $INFERENCES_PATH:/app/RoseTTAFold/RoseTTAFold-All-Atom/rf2aa/config/inference"

DOCKER_VOLUMES="$USER_CONFIGURED_PATH \
    $ROSETTAFOLD_ASSETS_PATH \
    $LSF_DOCKER_VOLUMES"

# BSUB
PATH="$PATH:/app/RoseTTAFold/mambaforge/condabin:/app/RoseTTAFold/mambaforge/bin" \
LSF_DOCKER_VOLUMES="$DOCKER_VOLUMES" \
bsub \
-n $SLOTS \
-M $MEMORY \
-R "rusage[mem=$MEMORY]" \
-R "gpuhost" \
-G $COMPUTE_GROUP \
-q $QUEUE \
-gpu 'num=1' \
-Is -a "docker($IMAGE_TAG)" /bin/bash
```

- Keep both scripts in the same directory.
- Change the variables such as the COMPUTE\_GROUP, OUTPUT\_PATH, INFERENCES\_PATH among others.
- E.g.

```
# SPECIFY ACCORDINGLY
COMPUTE_GROUP=compute-a.santiago

# where are the fasta or sfd files among others?
INPUT_PATH=$STORAGEN/rfaa/fastas/

# where do you want the resulting files?
OUTPUT_PATH=$STORAGEN/rfaa/outputs

# where are the config inference files?
INFERENCES_PATH=$HOME/rfaa/configs

# Required by LSF
QUEUE=qa
MEMORY=32GB
SLOTS=4
```

- Copy the base.yaml config file and put it in the config/inference directory.

```
job_name: "structure_prediction"
output_path: outputs/
checkpoint_path: checkpoint/RFAA_paper_weights.pt
database_params:
  sequencedb: ""
  hhdb: "pdb100_2021Mar03/pdb100_2021Mar03"
  command: make_msa.sh
  num_cpus: 4
  mem: 64
protein_inputs: null
na_inputs: null
sm_inputs: null
covale_inputs:  null
residue_replacement: null

chem_params:
  use_phospate_frames_for_NA: True
  use_cif_ordering_for_trp: True

loader_params:
  n_templ: 4
  MAXLAT: 128
  MAXSEQ: 1024
  MAXCYCLE: 4
  BLACK_HOLE_INIT: False
  seqid: 150.0


legacy_model_param:
  n_extra_block: 4
  n_main_block: 32
  n_ref_block: 4
  n_finetune_block: 0
  d_msa: 256
  d_msa_full: 64
  d_pair: 192
  d_templ: 64
  n_head_msa: 8
  n_head_pair: 6
  n_head_templ: 4
  d_hidden_templ: 64
  p_drop: 0.0
  use_chiral_l1: True
  use_lj_l1: True
  use_atom_frames: True
  recycling_type: "all"
  use_same_chain: True
  lj_lin: 0.75
  SE3_param:
    num_layers: 1
    num_channels: 32
    num_degrees: 2
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
    n_heads: 4
    div: 4
  SE3_ref_param:
    num_layers: 2
    num_channels: 32
    num_degrees: 2
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
    n_heads: 4
    div: 4
```

- Create a config inference yaml file.

  - The inference config file should inherit the base settings from the base.yaml file.
  - The defaults should be `base` and `_self_`.
  - Set the job name; otherwise, the default job name will be “structure\_prediction.”
  - All the input files should have be prefixed with inputs/.
  - For example, the K7N608.fasta file should be set as inputs/K7N608.fasta
  - Example config file named Vmn1r78\_E1050.rtaa.yaml

```
defaults:
  - base
  - _self_

job_name: "K7N608"

protein_inputs:
  A:
    fasta_file: inputs/K7N608.fasta

sm_inputs:
  B:
    input: inputs/66430.sdf
    input_type: "sdf"
```

- Launch the non-interactive app

```
# Usage ./bsub-non-interactive.sh <<inference config name>>
# Do not include the yaml file extention.

./bsub-non-interactive.sh Vmn1r78_E1050.rtaa
```

- Go to the OUTPUT\_PATH to find the results. A successful run should generate a pdb file.

```
drwx------+  Aug 26 11:16 2024-08-26
drwx------+  Aug 26 10:32 K7N608
-rw-------+  Aug 26 10:38 K7N608_aux.pt
-rw-------+  Aug 26 10:38 K7N608.pdb
```

- The output log should look like the following

```
Predicting:   0%|          | 0/1 [00:00<?, ?sequences/s]
Predicting: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 1/1 [00:00<00:00,  1.49sequences/s]
Predicting: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 1/1 [00:00<00:00,  1.48sequences/s]
Running HHblits against UniRef30 with E-value cutoff 1e-10
- 20:47:40.116 INFO: Input file = outputs/K7N608/A/hhblits/t000_.1e-10.a3m

- 20:47:40.116 INFO: Output file = outputs/K7N608/A/hhblits/t000_.1e-10.id90cov75.a3m

- 20:47:40.240 WARNING: Maximum number 100000 of sequences exceeded in file outputs/K7N608/A/hhblits/t000_.1e-10.a3m

- 20:48:50.358 INFO: Input file = outputs/K7N608/A/hhblits/t000_.1e-10.a3m

- 20:48:50.359 INFO: Output file = outputs/K7N608/A/hhblits/t000_.1e-10.id90cov50.a3m

- 20:48:50.646 WARNING: Maximum number 100000 of sequences exceeded in file outputs/K7N608/A/hhblits/t000_.1e-10.a3m

Running PSIPRED
Running hhsearch
Using the cif atom ordering for TRP.
./make_msa.sh inputs/K7N608.fasta outputs/K7N608/A 4 64  pdb100_2021Mar03/pdb100_2021Mar03
Aug 26 15:51:16 docker1_starter[2469871]: DEBUG:       run_docker returns 0
Aug 26 15:51:16 docker1_starter[2469871]: DEBUG:       Cleaning up temp files
Aug 26 15:51:16 docker1_starter[2469871]: DEBUG:       About to exit 0
```

# Available RoseTTAFold Versions

## Current Version:

- 1.0.0
