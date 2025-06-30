
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# RELION

- [Image Details](#image-details)
- [Requirements for Usage in RIS](#requirements-for-usage-in-ris)
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

- Docker image hosted at [ghcr.io/washu-it-ris/relion](http://ghcr.io/washu-it-ris/relion).
- Software Included:

  - Relion (<https://www3.mrc-lmb.cam.ac.uk/relion//index.php/Main_Page> )
  - CTFFIND (<https://grigoriefflab.umassmed.edu/ctffind4> )
  - GCTF (hhttps://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/zhang-software/)
  - MotionCor2 (<https://emcore.ucsf.edu/ucsf-motioncor2> )

# Requirements for Usage in RIS

- Interactions GUI sessions are done via the `Custom noVNC Image` application in Open On Demand (OOD).
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md).
- There are two fields beyond the basics that will need information specific to this image.

  - Environment Variables
  - Docker Image

## Environment Variables

- This information should be space separated in the field.

```
PASSWORD=password
```

- Optional variables

  - GUI display size. This can be changed with the following variables.

    - Width default: 1024
    - Height default: 768

  ```
  DISPLAY_WIDTH=<width> DISPLAY_HEIGHT=<height>
  ```

## Docker Image

```
gcr.io/ris-registry-shared/relion:<tag>
```

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.
- Once in an interactive relion session using the following command:

```
relion
```

# Available Versions

## Current Version:

- ghcr.io/washu-it-ris/relion

  - latest, 3.1
