
[Storage Platforms](../Storage%20Platforms.md)

# Recovering Data From Snapshots

- [Overview](#overview)
- [Manual Recovery Method](#manual-recovery-method)
- [Command Line Recovery Method](#command-line-recovery-method)
- [Globus Recovery Method for Storage1](#globus-recovery-method-for-storage1)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

> [!IMPORTANT]
> `.snapshots` vs `.snapshot`
>
> - The `.snapshots` nomenclature is used for Storage1.
> - The `.snapshot` nomenclature is used for Storage2.

# Overview

This document serves as a guide for current users with storage1 allocations who wish to recover data themselves from their `.snapshots` or `.snapshot` folder.

# Manual Recovery Method

> [!IMPORTANT]
> This method works better for small numbers of files or folders. If you are recovering a lot of data, it may be easier to use the command line instructions later in this document.

## Navigate to your storage drive.

- If you already have hidden files enabled, great! You can skip to step 3. Otherwise, your window will look similar to the window below.

### Mac Finder

![image-20250317-184416.png](../../attachments/1576538f-0f94-4f52-94b7-02b57a725776.png)

### Windows 10 File Explorer

![image-20250317-184434.png](../../attachments/4fa9170f-3d0e-4ac2-98c5-8063d48cc82f.png)

## Enable hidden files and folders

You will need to enable hidden files to see the `.snapshots` or `.snapshot` folder in your window.

To enable hidden files on a Mac, press `Command + Shift + Period`. Your Finder window should now look similar to the window below, with the `.snapshots` or `.snapshot` folder visible.

![image-20250317-184616.png](../../attachments/f8d6d1b4-ebdd-4769-a85b-0a29ac16486c.png)

To enable hidden files on Windows 10, navigate to the `Vie` tab and click the checkbox marked Hidden Items. The `.snapshots` or `.snapshot` folder should now be visible.

![image-20250317-184706.png](../../attachments/fd31229d-ac36-4ca0-9d15-bc3fcfa9e598.png)

## Restoring data from the `.snapshots` or `.snapshot` backup

Open the `.snapshots` or `.snapshot` folder. You should see folders named by day.

### Viewing the `.snapshots` or `.snapshot` folder on Mac

![image-20250317-184908.png](../../attachments/d6e00b17-8b0f-408c-86e1-5ba13821fe48.png)

### Viewing the `.snapshots` or `.snapshot` folder on Windows 10

![image-20250317-184916.png](../../attachments/8189feb3-ef13-45c3-99c5-9b76e36318d7.png)

Navigate to the folder corresponding with the most recent day before the files/folders were corrupted or deleted or overwritten, and then into the `Active` folder.

Select the files and folders you want to recover, then copy them (`Control + C` on Windows, `Command + C`on macOS).

Next, navigate back up to the main directory of your storage drive, and then into the Active folder. Paste the copied files (`Control + V` on Windows, `Command + V` on macOS).

Your files have been successfully restored!

# Command Line Recovery Method

## Windows

1. Open your storage drive and navigate to the `.snapshots` or `.snapshot` folder.
2. Locate the file or folder you want to restore, and copy the path name

   - (e.g. for Storage1 `\\storage1.ris.wustl.edu\tahan\.snapshots\Thu-tahan_active\Active\file.txt`).
   - (e.g. for Storage2 `\\storage2.ris.wustl.edu\tahan\.snapshot\Thu-tahan_active\Active\file.txt`).
3. Open Command Prompt.
4. For transferring a folder, type:

```
robocopy FOLDERPATH '\\storageN.ris.wustl.edu\DRIVENAME\Active\FOLDERNAME' /E /V /R:5 /W:5 /TBD /MT
```

5. Replace `FOLDERPATH` with the path to the folder from step 2, `DRIVENAME`

   with the name of your storage allocation (e.g. `tahan`) and `FOLDERNAME` with the name of the folder you are transferring.

> [!IMPORTANT]
> You can replace `/MT` with `/MT:#` (where # is a number from 1 to 128) to change the number of files robocopy processes simultaneously. `/MT` defaults to 8, so setting it to a number above 8 may increase transfer speed if you have a sufficiently powerful system and connection.

6. For transferring a file, type:

```
copy FILEPATH '\\storageN.ris.wustl.edu\DRIVENAME\Active'
```

7. Replace `FILEPATH` with the path to the file from step 2, and `DRIVENAME` with the name of your storage allocation (e.g. `tahan`).

## Mac

1. Open your storage drive and navigate to the .snapshots folder.
2. Locate the file or folder you want to restore, and copy the path name.

   - (e.g. for Storage1 `\\storage1.ris.wustl.edu\tahan\.snapshots\Thu-tahan_active\Active\file.txt`).
   - (e.g. for Storage2 `\\storage2.ris.wustl.edu\tahan\.snapshot\Thu-tahan_active\Active\file.txt`).
3. Open Terminal.
4. In Terminal, type:

```
rsync -avh PATH /Volumes/DRIVENAME/Active
```

5. Replace `PATH` with the path from step 2, and `DRIVENAME` with the name of your storage allocation (e.g. `tahan`).

# Globus Recovery Method for Storage1

Snapshots can also be recovered using Globus. Please see the Globus documentation for direction on accessing your storage allocation using the Globus interface. In the `PATH` field, enter in the path to your storage allocation, (e.g. `/storageN/fs1/${STORAGE_ALLOCATION}`). Enable the hidden files and folder by clicking the `Show Hidden Items` button, highlighted in red below.

![image-20250317-185215.png](../../attachments/820bd88e-872f-4ba2-9960-0fa82df873bd.png)

The `.snapshots` or `.snapshot` folder will now be visible.

![image-20250317-185238.png](../../attachments/b78692d3-90dd-4ff5-8986-4f6a7e3f8ada.png)

Navigate to the file or folder you’d like to recover. Click the `Transfer or Sync to...` button to open a new destination pane to the right. Naviage to the `Active` folder of your storage allocation. In the example below, a Python script will be restored to the `Active` folder.

![image-20250317-205619.png](../../attachments/ea338bc9-cd1a-48a3-828a-ccc96380196e.png)

Click the `Start` button to begin your transfer. Your transfer job is now submitted to Globus. You will receive an email once the transfer is complete and the restored file will be available in the `Active` folder.
