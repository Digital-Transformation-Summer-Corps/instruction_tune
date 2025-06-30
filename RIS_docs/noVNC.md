
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# noVNC

- [Image Details](#image-details)
- [Interactive GUI](#interactive-gui)
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

- Docker image hosted at <http://ghcr.io/washu-it-ris/novnc>
- Fluxbox (<http://www.fluxbox.org/> )
- noNVC (<https://kanaka.github.io/noVNC/> )
- supervisord ([http://supervisord.org](http://supervisord.org/))
- turboVNC (<https://turbovnc.org/> )
- websockify (<https://github.com/novnc/websockify> )
- x11vnc (<http://www.karlrunge.com/x11vnc/>)
- xterm (<http://invisible-island.net/xterm/> )
- Xvfb (<http://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml> )

# Interactive GUI

The [Open OnDemand](../../../Compute1/Open%20OnDemand.md) interface is the recommended way to run an application needing an interactive GUI.

The [Open OnDemand](../../../Compute1/Open%20OnDemand.md) page allows users to start various applications from basic templated forms. For all other applications the Custom noVNC Image form is recommended.

## Extending the noVNC Image

The RIS noVNC image can be extended for use on compute1 and the RIS Open OnDemand interface.

```
FROM ghcr.io/washu-it-ris/novnc:ubuntu22.04

RUN apt-get update && \
  apt-get install -y bison
```

## Run on Compute

- Interactions GUI sessions are done via the `Custom noVNC Image` application in Open On Demand (OOD).
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md) .
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
ghcr.io/washu-it-ris/novnc:<tag>
```

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.

# Available Versions

## Current Version:

- ghcr.io/washu-it-ris/novnc

  - ubuntu22.04
  - ubuntu22.04\_cuda12.4\_runtime
  - ubuntu22.04\_cuda12.4\_devel
