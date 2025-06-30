
[Compute1](../Compute1.md)

# Compute1 Quickstart

- [Connecting via SSH](#connecting-via-ssh)
- [Connecting to Open On Demand (OOD)](#connecting-to-open-on-demand-ood)
- [Starting Up an Application in OOD](#starting-up-an-application-in-ood)
- [Navigating Your Job](#navigating-your-job)
- [Using the Compute RIS Desktop to Access the THPC Environment](#using-the-compute-ris-desktop-to-access-the-thpc-environment)
- [Using the Custom noVNC Image to Load Additional or Personalized Software](#using-the-custom-novnc-image-to-load-additional-or-personalized-software)
- [Connect Using Local VNC Viewer](#connect-using-local-vnc-viewer)
- [Further Information about the Compute1 Platform](#further-information-about-the-compute1-platform)

> [!IMPORTANT]
> **Compute Resources**
>
> - Have questions or need help with compute, including activation or issues? Follow [this link.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43)
> - [RIS Services Policies](../RIS%20Services%20Policies.md)

> [!IMPORTANT]
> **storageN**
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Connecting via SSH

- If you are off campus, you will need to use a VPN to access compute1.
- Instructions for accessing the WashU VPNs can be found here: <https://it.wustl.edu/items/connect/>
- If you run into issues using the VPN, you will need to follow the directions in the previous link to contact WashU IT proper.

> [!IMPORTANT]
> WashU has several VPNs. compute1 can be accessed from the following VPNs:
>
> - `msvpn.wusm.wustl.edu`
> - `danforthvpn.wustl.edu`

- If you are not familiar with a command line interface or would prefer a GUI, you only need to connect via SSH once.
- You can use SSH to connect to the compute platform with your WashU Key (single sign on) and password.
- E.g.

```
ssh washukey@compute1-client-1.ris.wustl.edu
```

![](../../attachments/803e00b3-6b36-42e6-aeec-3ef1912bb79e.gif)
> [!IMPORTANT]
> **Where to SSH From**
>
> - If you are using Windows, you can use SSH from either the command prompt or powershell if you have Windows 10 or 11.
> - Or you can use a software like PuTTY or MobaXterm .
> - If you use a Mac or Linux, you can use the built in terminal to connect via SSH.

# Connecting to Open On Demand (OOD)

- OOD is a web browser based interface that connects to compute resources, in this case, the RIS Compute Platform.
- If you have logged in via SSH **at least once**, you can point your browser to <http://ood.ris.wustl.edu> .
- Once you’re logged into OOD you can:

  - Start up one of the applications installed (Jupyter, Matlab, Rstudio, Relion3,Stata)
  - Use one of the RIS noVNC created Docker images (found here) via the Custom noVNC Image application.
  - Start up the Compute RIS Desktop to use a more traditional HPC environment via the module load system. [THPC](../Applications/Compute1%20Applications/RIS%20Developed%20Containers/THPC.md).

![](../../attachments/a0332ee9-e338-406d-a10f-a4187fb175e0.png)

# Starting Up an Application in OOD

- All of the applications, including the Compute RIS Desktop and Custome noVNC image, have the same base fields.

![image-20250218-163139.png](../../attachments/dab8575e-ccc9-435a-a1ba-77ad00dcb1ad.png)![image-20250218-164028.png](../../attachments/45f012f0-1fbe-4089-a2ae-8659d48472d9.png)

- The base fields are as follows:

  - Mounts

    - This is where you input your storage allocation information.
    - The format for this is based of the linux format and looks like: `/storageN/fs1/${STORAGE_ALLOCATION}/Active:/storageN/fs1/{$STORAGE_ALLOCATION}/Active`
    - `${STORAGE_ALLOCATION` should be replaced with the name of your storage allocation.
    - The name before the colon is the full path of your storage allocation.
    - The name after the colon is what you would like the path to be named in the application. (We recommend leaving it the same.)
    - More information on storage allocation [Access Storage Volumes](Access%20Storage%20Volumes.md).
  - Job Group

    - The default Job Group should be something like washukey/ood
    - This field should be automatically filled in and does not need to be changed.
    - If you wish to know more about job groups, [Job Execution Examples](Job%20Execution%20Examples.md).
  - User Group

    - This is a drop down menu of the compute groups a user is part of.
    - This does not need to be changed if the user is a member of a single group.
  - Queue

    - This is a field where the user can input which queue they would like to use.
    - Only queues that the user is part of are valid. If you are not part of a queue, you job will error.
    - All restrictions on a queue are still in effect, even if the job is launched through OOD.
  - SLA Name

    - This field is for users who are part of a subscription tier and can be ignored by general and condo users.
    - More information about subscriptions and SLA can be found here.
  - Memory (GB)

    - This is the amount of memory or RAM you want to give the job.
    - This is constrained by the selected queue and the amount of physical memory available on hosts in that queue.
  - Number of hours

    - This is the number of hours you want your job to run.
    - This is limited by the queue chosen.

      - general queue limit is 28 days or 672 hours
      - general-interactive queue limit is 24 hours
  - GPUs to Allocate

    - This is the number of GPUs you wish to have available for your job.
    - This is limited by the number of GPUs available to GPU hosts in the chosen queue (typically 4).
    - You can find more [Job Execution Examples](Job%20Execution%20Examples.md).
  - Number of processors

    - This is the number of processors or CPUS you wish to have available for you job.
    - This is limited by the number of CPUs available to hosts in the chosen queue (varies).
    - You can find more [Job Execution Examples](Job%20Execution%20Examples.md).
  - Font Size

    - This is a drop down menu that you can use to select how big you want the font in your job.
- Some applications have fields that are specific to them. They will have a description of what’s required for that field.
- Once you have filled in the appropriate fields, you click on the launch button and your job will be launched.

# Navigating Your Job

- Once your job is launched you will be taken to the My Interactive Sessions part of OOD.
- Here you can see all of the sessions you have running.
- There are 3 states to a job.

  - Queued

    ![image-20250218-164251.png](../../attachments/db978276-a030-4689-9131-50e9a7cc7a2d.png)
  - Starting

    ![image-20250218-164315.png](../../attachments/8210bfcc-1504-413b-ac37-10a5654e4d3c.png)
  - Running

    ![image-20250218-164340.png](../../attachments/afac813a-2487-470e-838b-77b5c98e5346.png)
- Once a job is in the Running state, you will have a launch button available. Clicking this will launch you into your chosen application.
- The interface you see will be different based on the application you use, though most will launch you into a linux desktop.
- If you wish to know more about OOD, [Open OnDemand](Open%20OnDemand.md)

# Using the Compute RIS Desktop to Access the THPC Environment

- The THPC environment uses Lmod to dynamically control the environment through use of module files.
- Here are the official docs for guidance beyond these basic commands.
- Shorthand exists for these module commands as well. Use ml -h in a job session.
- Loading modules without specifying a version will result in the default module being loaded.
- Default module versions are denoted with (D) in the listing if more than one version is available.
- To view what modules are available you use module avail via the terminal.

![image-20250218-164437.png](../../attachments/3619f66b-a04a-4e52-8513-053397671428.png)

- To list what modules are currently loaded you use module list.
- To load a module you use module load package where package is the name of the module.
- To unload a module you use module unload package.

![image-20250218-164510.png](../../attachments/630afd70-05a9-4f1d-ae0f-386c9e236455.png)

# Using the Custom noVNC Image to Load Additional or Personalized Software

- This application does not have a preset Docker image, the user supplies one.
- The list for software developed to be used this way, can be found here.
- If you create a Docker image using these images as a base, you can use that container here as well.
- There are two additional fields in this application that are important.

  - Environment Variables

    - This is where you put environment variables that the image needs to run.
    - E.g.

      - PASSWORD=password
      - PATH=/opt/conda/bin/:$PATH
    - Necessary variables for an image are included in their individual documentation pages.
  - Docker Image

    - This is where you put in the name of the Docker image you wish to use, including any tags.
    - E.g.

      - <http://gcr.io/ris-registry-shared/matlab>
      - <http://ghcr.io/washu-it-ris/rstudio:4.3.0>

![image-20250218-164831.png](../../attachments/4fc82910-1cbb-4b2f-bbec-519b33e84943.png)

# Connect Using Local VNC Viewer

- [Local VNC Viewer](Local%20VNC%20Viewer.md)

# Further Information about the Compute1 Platform

- If you wish to know more about using the Compute Platform and the command line interface:

  - Check out our [FAQ](../FAQ.md).
  - Check out our [Compute1](../Compute1.md).
  - Check out our [Compute1](../Compute1.md) and [Compute1](../Compute1.md) docs.
