
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Huygens

- [Image Details](#image-details)
- [Interactive GUI Session](#interactive-gui-session)
- [Interactive Command-Line Session](#interactive-command-line-session)
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

- huygenspro - <https://svi.nl/Huygens-Professional>
- hucore (command line interface) - <https://svi.nl/Huygens-Core> (This is the only command that will run in the CLI without needing a GUI/display)
- essential - <https://svi.nl/Huygens-Essential>
- huygensloc - <https://svi.nl/Huygens-Localizer>

# Interactive GUI Session

- Interactions GUI sessions are done via the Custom noVNC Image application in Open On Demand (OOD).
- You can find out more about OOD here: Compute Quick Start.
- The Docker Image field should be filled in with the information below.

```
ghcr.io/washu-it-ris/huygens:<tag>
```

> [!IMPORTANT]
> Huygens Docker Tag:
>
> The `<tag>` will refer to the version of Huygens in the Docker container.

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.
- Once in an interactive session use the following commands in the terminal, depending on which version you want.

## Huygens Professional

```
> huygenspro
```

## Huygens Essential

```
> essential
```

## Huygens Localizer

```
> huygensloc
```

- The terminal command and GUI should look like the following.

![image-20250313-143209.png](../../../../attachments/f0e4f5b7-173b-406c-bc0a-afa7dbfcd175.png)
> [!IMPORTANT]
> License Information
>
> - If no trial license or purchased license available, software will run In freeware version.
> - Per SVI support team, manual login is required unless a node-locked license string is properly set up.
> - This functionality does not have a universal solution for the RIS community/cluster at this
> - **To attempt this solution, license location should be mounted as follows:**
>
>   - `/path/to/huygens_license_folder/:/opt/huygens/huygensLicense`
>   - where the license file is named `huygensLicense`

# Interactive Command-Line Session

- If you wish to use Huygens in an interactive command-line session, you can do so with the following commands.

```
> bsub -Is -a 'docker(ghcr.io/washu-it-ris/huygens:<tag>)' -q general-interactive /bin/bash
```

- Once in the container, you can run the command line version of Huygens with the following command.

```
> hucore
```

# Available Versions

### Current Versions

- `ghcr.io/washu-it-ris/huygens`

  - 24.10
