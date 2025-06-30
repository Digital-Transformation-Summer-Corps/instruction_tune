---
tags:
  - '#compute1'
  - '#vscode'
---

[Compute1](../Compute1.md)

# Visual Studio Code on Compute1

- [Connecting to Compute1](#connecting-to-compute1)
- [Opening a Storage Directory](#opening-a-storage-directory)
- [Testing Code](#testing-code)

This documentation goes over how to connect to Compute1 through Visual Studio Code (VSC) for code editing and development purposes.

VSC can be found here: <https://code.visualstudio.com/>

VSC is a rather expansive editor that has many features and extensions that you can add to help customize your experience.

VSC is also free and open source and has Mac OS, Windows, and Linux versions available.

> [!WARNING]
> There is a known issue with the newer versions of VSC (1.101+) please use version 1.100. It can be downloaded here: <https://code.visualstudio.com/updates/v1_100>

# Connecting to Compute1

- In order to connect to Compute1, an extension will need to be installed that enables an ssh connection to be established.

![](../../attachments/0169d3d3-3787-44fa-8795-3b7320be1eba.png)

- “Remote - SSH” is the recommended extension for this. Simply click the Install button and VSC will install the extension.
- Once “Remote - SSH” has been installed, to connect, click the double carrot icon in the bottom left corner of the window.

![](../../attachments/cff5fa51-69e6-4615-b2b2-23177c591037.png)

- After clicking the icon, it will open a new window and at the middle of the top of the window, there will be box that, when selected, will give a dropdown menu.
- Select, `Connect to Host...`

![](../../attachments/7fbe6ffe-870a-44da-bcd9-726b91d59705.png)

- This will open the option to connect to a host.

## First Time Connecting

- Click on the `Add New SSH Host…`.

![](../../attachments/971cd4a2-36de-402d-ab21-a95c3d6dd327.png)

- Enter the information for one of the Compute1 login clients.

```
ssh washukey@compute1-client-N.ris.wustl.edu
```

![](../../attachments/05475478-cd31-4229-b8f9-32a34bddbc43.png)

- It will ask to select an SSH configuration file to use, select the top one (local config).

![](../../attachments/d22b68fe-67ea-4a64-85be-8b001e8fe2f2.png)

- A pop box will appear in the bottom right corner, click the Connect button.

![](../../attachments/0156fac4-e6be-4623-a447-03d1a119cdff.png)

- It will then prompt for a password. Use the password associated with the WashU key.

![](../../attachments/388b5dfd-572a-4e2f-89be-c9be8f9c4996.png)

- Once connected, it will default to a screen without any directories open.

![](../../attachments/5c4ad943-ee03-4a4c-9977-06e631a194e0.png)

## Connecting to an Established Host

- Click on an established login client to use to connect.

![](../../attachments/971cd4a2-36de-402d-ab21-a95c3d6dd327.png)

- It will then prompt for a password. Use the password associated with the WashU key.

![](../../attachments/388b5dfd-572a-4e2f-89be-c9be8f9c4996.png)

- Once connected, it will default to a screen without any directories open.

![](../../attachments/5c4ad943-ee03-4a4c-9977-06e631a194e0.png)

# Opening a Storage Directory

- On the left side of the window there will be an “Open Folder” button. To open a folder or directory, click on this button.

![](../../attachments/5c928fdf-d241-4b21-ac24-901a95373d7f.png)

- It will ask what folder to open. This defaults to the home directory on Compute1, but any Storage allocation can be used.

![](../../attachments/f6423ff6-9504-44b2-ba1c-db5faab773d9.png)

- Once the Storage allocation has been selected, files can be opened and edited.

![](../../attachments/7bf6180d-e862-4a3b-bf83-6d4303f9af4c.png)

# Testing Code

- In order to test the code or scripts that are being developed, a terminal session will need to be opened.
- This can be done via the icons in the top right of the window.
- Click on the 2nd from the right icon to open a terminal.

![](../../attachments/14e6278a-c842-43b0-a88c-9b959fbf6ba6.png)![](../../attachments/8f7963ee-cef3-4099-a3c1-426592f87342.png)

- Once the terminal is open, the code can be tested.

> [!WARNING]
> Just like when using software or code for analysis, users need to launch a job to run tests for development.

- Start up an interactive job via a bsub command using a Docker image that has the software or development environment needed for the code being developed.

![](../../attachments/684fb9c9-bce0-48ce-9e8e-1e7f64a78b13.png)

- Code and software can now be edited and tested using Compute1 resources.
