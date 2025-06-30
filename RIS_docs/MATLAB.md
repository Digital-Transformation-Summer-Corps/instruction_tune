
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# MATLAB

- [Quick Start](#quick-start)
  - [Interactive GUI Session](#interactive-gui-session)
  - [Interactive Command-Line Session](#interactive-command-line-session)

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

# Quick Start

## Interactive GUI Session

- Interactive GUI sessions are done via Open On Demand (OOD).

  - You can use the `Matlab` application.
  - You can also use the `Compute RIS Desktop` application.
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md).
- Fill out the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.
- If you use the `Matlab` application, Matlab will be started up automatically.
- If you use the `Compute RIS Desktop` application, please follow the steps below.

  - Once in an interactive session using the following commands to load and launch MATLAB.

  ```
  module load MATLAB
  matlab
  ```

  - Versions available can be seen via the following command.

  ```
  module avail MATLAB
  ```

## Interactive Command-Line Session

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with `-G group_name` or by setting the `LSB_SUB_USER_GROUP` variable.

- If you wish to use MATLAB in an interactive command-line session, you can do so via the THPC terminal.
- You can find out about using the THPC terminal [THPC](THPC.md)
- You load Matlab just like in the OOD version.

```
module load MATLAB
```
