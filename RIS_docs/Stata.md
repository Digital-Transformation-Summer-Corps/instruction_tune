
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Stata

- [Image Details](#image-details)
- [Requirements for Usage in RIS](#requirements-for-usage-in-ris)
- [Interactive GUI Session](#interactive-gui-session)
- [Interactive Command-Line Session](#interactive-command-line-session)
- [Generating a Stata License File](#generating-a-stata-license-file)
- [Multiple Stata Licenses](#multiple-stata-licenses)

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

- Docker image hosted at http://ghcr.io/washu-it-ris/stata
- Stata for Linux (<https://www.stata.com/> )

# Requirements for Usage in RIS

- Use of this software requires a valid Stata license.

  - The license will be in the form of a license file named `stata.lic`. This file can also be generated from the “License and Activation Key” form Stata provides. Please see the [Generating a Stata License File](https://washu.atlassian.net/wiki/spaces/RUD/pages/edit-v2/1782645003#Generating-a-Stata-License-File) section for more information.
  - It is recommended to store the license in your storage allocation. (e.g. `/storageN/ fs1/${STORAGE_ALLOCATION}/Active/Stata_License`). Please do not store any other files in the folder containing the Stata license.

# Interactive GUI Session

Please use the following commands to submit an interactive GUI session for Stata.

> [!IMPORTANT]
> - Complete export PASSWORD= with that of your choosing. Do not leave this blank.
> - Replace the example Stata license folder path (`/storageN/fs1/${STORAGE_ALLOCATION}/Active/ Stata_License`) with the path to the folder containing your Stata license.

- Interactions GUI sessions are done via the `Custom noVNC Image` application in Open On Demand (OOD).
- You can find out more about OOD here: [Compute1 Quickstart](../../../Compute1/Compute1%20Quickstart.md).
- There are three fields beyond the basics that will need information specific to this image.

  - Mounts
  - Environment Variables
  - Docker Image

## Mounts

- You need to include the following in your `Mounts` field in addition to your other inputs.

```
/storageN/fs1/${STORAGE_ALLOCATION}/Active/Stata_License:/opt/stata
```

## Environment Variables

- This information should be space separated in the field.

```
PASSWORD=password PATH=/app/stata:$PATH
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
ghcr.io/washu-it-ris/stata:18
```

- Fill out the rest of the fields with the appropriate information (explained in the quick start).
- Launch the job through the methods described in the quick start.
- Open Stata by entering the version of Stata you are licensed to use into the GUI terminal.

  - This can be `xstata`, `xstata-mp` or `xstata-se`.
  - Information on the differences between each Stata version can be found at <https://www.stata.com/products/which-stata-is-right-for-me/>

# Interactive Command-Line Session

- If you wish to use Stata in an interactive command-line session, you can do so with the following commands. Please see the note above with information on mounting a folder containing your Stata license.

```
export PATH=/app/stata:$PATH
export LSF_DOCKER_VOLUMES="/storageN/fs1/${STORAGE_ALLOCATION}/Active/Stata_License:/opt/stata"
bsub -Is -q general-interactive -a 'docker(ghcr.io/washu-it-ris/stata:18)' /bin/bash
```

# Generating a Stata License File

If you do not have a Stata License file (`stata.lic`), you can generate the license using the “License and Activation Key” form provided by Stata after license purchase.

1. Obtain Serial Number, Code and Authorization from Stata via a license purchase.
2. Submit a Stata interactive command-line session.
3. Navigate to the Stata application folder

```
cd /app/stata
```

4. Run the Stata license generator

```
./stinit
```

5. Fill out the information requested by the license generator, including the Serial Number, Code and Authorization in the document provided by Stata after license purchase.
6. The `stata.lic` is now generated and available in the `/app/stata` folder. To avoid having to repeat this process, copy the license to a secure location, such as a storage allocation. The secured license file can then be used in subsequent interactive Stata jobs. Please see the note above for more information.

# Multiple Stata Licenses

If you are licensed to use more than one version of Stata supported by RIS, then it is important to mount the folder containing the license pertaining to the version of Stata you will be using.

A suggestion would be to create a folder in your storage allocation for each supported version of Stata you are licensed to use. Name the folder after the version of Stata and copy the corresponding license to the folder. As recommended earlier, please be sure that no other files are in the folder containing the Stata license.

An example folder containing the Stata 18 license would be:

```
/storageN/fs1/${STORAGE_ALLOCATION}/Active/Stata18_License/
```
