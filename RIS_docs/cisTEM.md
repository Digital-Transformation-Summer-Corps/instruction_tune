
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# cisTEM

- [Software Included](#software-included)
- [Development Notes](#development-notes)
- [Dockerfile](#dockerfile)
- [Requirements for Usage in RIS](#requirements-for-usage-in-ris)
- [Docker Image](#docker-image)

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

- cisTem (<https://cistem.org/software> )

# Development Notes

- The official cisTEM documentation can be [found here.](https://cistem.org/system/tdf/uploads/cisTEM_tutorial.pdf?file=1&type=cistem_details&id=7&force=)

# Dockerfile

- The cisTEM image is hosted at the following location.

```
gcr.io/ris-registry-shared/cistem
```

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

# Docker Image

```
gcr.io/ris-registry-shared/cistem
```

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.
- Once in an interactive cisTEM session using the following command:

```
cisTEM
```

![image-20250313-141049.png](../../../../attachments/8d3d9fb2-2daa-4c97-8ac2-825f344dceed.png)![image-20250313-141105.png](../../../../attachments/b1db8835-c046-47f2-b259-2368ecd8f375.png)
