
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# TensorFlow

- [Software Included](#software-included)
- [Interactive GUI Session](#interactive-gui-session)
- [Extend the TensorFlow Image](#extend-the-tensorflow-image)
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

# Software Included

- noVNC (<https://novnc.com/info.html> )
- Python ([https://www.python.org](https://www.python.org/))
- TensorFlow ([https://www.tensorflow.org](https://www.tensorflow.org/))

# Interactive GUI Session

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
ghcr.io/washu-it-ris/tensorflow:<tag>
```

> [!IMPORTANT]
> TensorFlow Docker Tag:
>
> The `<tag>` will refer to the version of Tensorflow in the Docker container. Please see below for a current list of supported TensorFlow versions and their corresponding tags.

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- You will need to select a GPU in the field for doing so since TensorFlow uses GPUs.
- Launch the job through the methods described in the quick start.
- Once in an interactive TensorFlow session using the following command:

```
> ipython -i --no-banner
```

## Interactive Command-Line Session

- If you wish to use TensorFlow in an interactive command-line session, you can do so with the following command.

```
bsub -Is -R 'gpuhost' -gpu "num=1" -q general-interactive -a 'docker(ghcr.io/washu-it-ris/tensorflow:<tag>)' /bin/bash
```

# Extend the TensorFlow Image

You may wish to extend the TensorFlow container with additional Python packages. Below is an example Dockerfile which extends the latest RIS-hosted TensorFlow image with the following packages:

- pandas (<https://pandas.pydata.org/> )
- matplotlib (<https://matplotlib.org/> )

## Dockerfile

```
FROM ghcr.io/washu-it-ris/tensorflow:latest
RUN pip install pandas matplotlib
```

# Available Versions

## Current Version:

- ghcr.io/washu-it-ris/tensorflow

  - latest, 24.04-tf2-py3
