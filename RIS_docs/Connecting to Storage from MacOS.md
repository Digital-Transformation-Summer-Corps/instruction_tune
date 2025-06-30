
[Storage Platforms](../Storage%20Platforms.md)

# Connecting to Storage from MacOS

- [Step-by-step Guide](#step-by-step-guide)
- [If your sidebar does not have “Locations”:](#if-your-sidebar-does-not-have-locations)
- [If storage devices are not visible on your Desktop](#if-storage-devices-are-not-visible-on-your-desktop)
- [Making Desktop shortcuts permanent](#making-desktop-shortcuts-permanent)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Step-by-step Guide

Disable the writing of .DS\_Store files to network mounts by running this command from the Terminal app:

```
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
```

> [!WARNING]
> Note that this applies to all network mounts, not just RIS SMB.

Create or update the `/etc/nsmb.conf` file. This can be done with a text editor or by running this command from the Terminal which will create the `/etc/nsmb.conf` file if it does not already exist:

```
sudo tee /etc/nsmb.conf <<EOF
[default]
smb_neg=smb2_only
dir_cache_off=yes
notify_off=yes
soft=yes
streams=yes
file_ids_off=yes
EOF
```

The use of sudo means you will need administrative privileges on this Mac.

Synchronize the SMB config, again from the Terminal:

```
/usr/libexec/smb-sync-preferences
```

Click in the background area of the desktop. This will put you in Finder. In the Finder menu at the top left of the screen, click “Go”:

![image-20250317-143550.png](../../attachments/ff2b8b46-8898-4ff2-890e-0ada48f4a62c.png)

From the drop-down menu, click on “Connect to Server”. (⌘ K)

![image-20250317-143604.png](../../attachments/2b86580b-6ca0-43ab-ae50-dd0fbfb2cf5e.png)

Use the “Connect to Server” option on the menu drop-down. Once the “Connect to Server” dialog appears, use `smb://storageN.ris.wustl.edu/PI WashU Key ID` as the server address, in this example `juliesmith`:

![image-20250317-143635.png](../../attachments/6193f4a7-0998-49aa-9e99-469a1c7c2d6d.png)

You will be prompted for your WashU Key credentials (you will enter your Washu Key ID and password):

![image-20250317-143712.png](../../attachments/7bdf3aea-a4ff-4d72-8505-0ac4e560a36e.png)

Once you have completed these steps you will be presented with a finder window showing the Research Drive and all the folders you have access to:

![image-20250317-143737.png](../../attachments/fae87c4c-a725-45ec-ba13-16e1d6221080.png)

# If your sidebar does not have “Locations”:

If you cannot see the Research Drive under “Locations” on the sidebar in Finder (circled above), click on “File” in the top left corner:

![image-20250317-143853.png](../../attachments/e120c79f-7fc3-4c58-96f3-d6ffb3b75930.png)

From the drop-down menu, select “Add to sidebar”. The Research Drive should now appear in your sidebar. If it doesn’t, you may have to restart:

![image-20250317-143923.png](../../attachments/c6bf165a-bb80-4d75-85b5-6df986ef507e.png)

In addition to the shortcut on the sidebar, there should also be a shortcut to the Research Drive on your desktop.

![image-20250317-144042.png](../../attachments/01b98c6e-bfc6-42f5-a0c2-189a60f42efc.png)

# If storage devices are not visible on your Desktop

If storage volumes are not visible on your Desktop, click on the desktop background to go to Finder, and click on “Finder” in the top left corner of the screen.

![image-20250317-144124.png](../../attachments/4648304b-6848-4315-a878-64827f42c975.png)

From the drop-down menu, click on “Preferences”.

![image-20250317-144135.png](../../attachments/8fa21c1d-17f2-4300-aef3-3a8fdb7f392e.png)

In the Preferences menu, under the “General” tab, check the box next to “Connected servers”. The shortcut should now appear on your desktop.

![image-20250317-144153.png](../../attachments/236e1ae1-0f1d-4dc5-a0d1-91fe4d96f743.png)

# Making Desktop shortcuts permanent

This shortcut, unfortunately, is not permanent and will disappear if you log out, restart, or lose Internet connection. To make it permanent, follow these steps.

1. Open System Preferences from the Dock. If it is not in your Dock, search for “System Preferences” using the search in the top right of the screen:

![image-20250317-144239.png](../../attachments/eb9a7828-2433-4675-96da-fdc2b9e23520.png)

2. Click on “Users and Groups”.

![image-20250317-144312.png](../../attachments/9d133a81-6a25-495f-9225-220a6dab3040.png)

3. Click on “Login Items”.

![image-20250317-144329.png](../../attachments/ec9bb269-0d21-44b6-8c8e-bcab8ab9a570.png)

4. Click on the “+” at the bottom left of the tab.

![image-20250317-144348.png](../../attachments/91c92720-6b2f-475b-85e8-24eed3dde14a.png)

5. In the window that pops up, navigate to your Research Drive using the sidebar, and then press “Add”.

![image-20250317-144409.png](../../attachments/7fd768ce-ac42-404d-94a3-29b1b01b0f73.png)

6. The list in Login Items should now show the drive.

![image-20250317-144432.png](../../attachments/4ee1c25d-cc31-4e8f-87a7-cc38c8a91f61.png)

The drive and the shortcut should now be permanently connected so long as you have a network connection. You will have to re-enter your credentials if you restart the computer, but this can be connected to your login keychain (and thus done automatically) if you press “Always Allow” when prompted for your username and password, or by checking a box for “Remember this password in my keychain”.
