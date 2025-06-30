
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# RFdiffusion

- [Image Details](#image-details)
- [Using RFdiffusion with RIS](#using-rfdiffusion-with-ris)
- [Available Versions](#available-versions)

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

# Image Details

- Docker image hosted at `ghcr.io/washu-it-ris/rfdiffusion` .
- Software Included:

  - RFdiffusion (<https://github.com/RosettaCommons/RFdiffusion/tree/main> )

# Using RFdiffusion with RIS

- Interactions GUI sessions are done via the `Custom noVNC Image` application in Open OnDemand (OOD).
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md).
- There are two fields beyond the basics that will need information specific to this image.

  - Mounts
  - Docker Image

## Environment Variables

- This information should be space separated in the field that says “mounts” on the Open OnDemand Portal.

```
/scratch1/fs1/ris/references/RFdiffusion/:/scratch1/fs1/ris/references/RFdiffusion/
```

## Docker Image

```
ghcr.io/washu-it-ris/rfdiffusion:<tag>
```

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job using the submit button as described in the quick start.
- Once in an interactive relion session using the following command:

## Model Availability

- We have provided the model files at /scratch1/fs1/ris/references/RFdiffusion/models/ so the users need not download these again. These models were downloaded using the [script](https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/download_models.sh) provided in the RFdiffusion source code that shows what files to download. As of 03/13/2024 the following model files (with hashes) are downloaded and provided:

```
wget -P /scratch1/fs1/ris/references/RFdiffusion/models/ \
http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt \
http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt
```

- To use the provided models please add the above path to inference.model\_directory\_path configuration.
- RFdiffusion may not run on thpc unless provided with a schedule path with the inference.schedule\_directory\_path config option. Please note: The mentioned path does not appear to need anything in it to run, but RIS is not support for RFdiffusion and cannot speculate on the config setting’s use beyond its need to be explicitly defined in this context.

## Batch Job Examples

- Refer below is an example bsub job:

```
LSF_DOCKER_VOLUMES="$LSF_DOCKER_VOLUMES /scratch1/fs1/ris/references/RFdiffusion/:/scratch1/fs1/ris/references/RFdiffusion/" \
bsub \
-n 4 \
-M 32GB \
-R rusage[mem=32GB] \
-R select[cpumicro=some_arch] \
-G <your_group> \
-q general \
-a 'docker(ghcr.io/washu-it-ris/rfdiffusion:latest)' \
python3.9 /app/RFdiffusion/scripts/run_inference.py \
inference.output_prefix=$HOME/outputs/motifscaffolding \
inference.model_directory_path=/scratch1/fs1/ris/references/RFdiffusion/models \
inference.input_pdb=/scratch1/fs1/ris/references/RFdiffusion/inputs/5TPN.pdb \
inference.num_designs=3 \
inference.schedule_directory_path=$HOME/schedules \
contigmap.contigs='[10-40/A163-181/10-40]' \
```

- To use GPUs change the bsub arguments:

```
LSF_DOCKER_VOLUMES="$LSF_DOCKER_VOLUMES /scratch1/fs1/ris/references/RFdiffusion/:/scratch1/fs1/ris/references/RFdiffusion/" \
bsub \
-n 4 \
-M 32GB \
-R 'gpuhost' \
-G <your_group> \
-q general \
-gpu 'num=1' \
-oo path/to/output \
-Is -a 'docker(ghcr.io/washu-it-ris/rfdiffusion:test)' \
python3.9 /app/RFdiffusion/scripts/run_inference.py \
inference.output_prefix=$HOME/outputs/motifscaffolding \
inference.model_directory_path=/scratch1/fs1/ris/references/RFdiffusion/models \
inference.input_pdb=/scratch1/fs1/ris/references/RFdiffusion/inputs/5TPN.pdb \
inference.num_designs=3 \
inference.schedule_directory_path=$HOME/schedules \
contigmap.contigs='[10-40/A163-181/10-40]'
```

- Please do not forget to change <your\_group> with your compute allocation group.
- Update the cpumicro architecture ‘some\_arch’ line from: cascadelake, skylake, zen2, etc.

# Available Versions

## Current Version:

- ghcr.io/washu-it-ris/rfdiffusion

  - latest
