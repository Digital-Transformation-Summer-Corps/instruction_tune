
[Storage Platforms](../Storage%20Platforms.md)

# Moving Data With Rclone

- [What is Rclone?](#what-is-rclone)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building an Endpoint](#building-an-endpoint)
- [Use Case](#use-case)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# What is Rclone?

From <https://hub.docker.com/r/rclone/rclone>:

> Rclone (“rsync for cloud storage”) is a command line program
>
> to sync files and directories to and from different cloud storage providers.

# Overview

You will install rclone on your local computer. Through the command rclone config, you will create a credential file for rclone to connect to your WashU Box on your local computer. By copying the file to your home directory on RIS compute1 client, you will be able to access your Box storage through your rclone container on a compute1 exec node.

# Prerequisites

1. A WashU Box account
2. A user account for RIS storageN and compute1 services

# Building an Endpoint

## Installation

- For macOS users, run the following command to install rclone with Homebrew.

  ```
  > brew install rclone
  ```
- For Windows users, download the relevant archive file from <https://rclone.org/downloads/> for your environment. Then, extract the rclone.exe binary from the archive.
- For Linux/BSD users, run the following command to install rclone.

  ```
  > curl https://rclone.org/install.sh | sudo bash
  ```

## Configuration

### Creating the configuration file for the connection to WUSTL Box

- Open a terminal where rclone has been installed
- Run `rclone config` to start the interactive process.

```
> rclone config
No remotes found - make a new one
n) New remote
s) Set configuration password
q) Quit config
n/s/q> 
```

- Type n to setup a new remote connection. It will ask for the name for your new remote connection.

```
n/s/q> n
name>
```

- Type Box for example, as the name of your new remote connection. It will ask for the storage type.

```
name> Box
Type of storage to configure.
Enter a string value. Press Enter for the default ("").
Choose a number from below, or type in your own value
```

- Type box for the storage type.

```
Type of storage to configure.
Enter a string value. Press Enter for the default ("").
Choose a number from below, or type in your own value
 1 / 1Fichier
   \ "fichier"
 2 / Alias for an existing remote
   \ "alias"
 3 / Amazon Drive
   \ "amazon cloud drive"
 4 / Amazon S3 Compliant Storage Provider (AWS, Alibaba, Ceph, Digital Ocean, Dreamhost, IBM COS, Minio, Tencent COS, etc)
   \ "s3"
 5 / Backblaze B2
   \ "b2"
 6 / Box
   \ "box"
 7 / Cache a remote
   \ "cache"
 8 / Citrix Sharefile
   \ "sharefile"
 9 / Dropbox
   \ "dropbox"
10 / Encrypt/Decrypt a remote
   \ "crypt"
11 / FTP Connection
   \ "ftp"
12 / Google Cloud Storage (this is not Google Drive)
   \ "google cloud storage"
13 / Google Drive
   \ "drive"
14 / Google Photos
   \ "google photos"
15 / Hubic
   \ "hubic"
16 / In memory object storage system.
   \ "memory"
17 / Jottacloud
   \ "jottacloud"
18 / Koofr
   \ "koofr"
19 / Local Disk
   \ "local"
20 / Mail.ru Cloud
   \ "mailru"
21 / Mega
   \ "mega"
22 / Microsoft Azure Blob Storage
   \ "azureblob"
23 / Microsoft OneDrive
   \ "onedrive"
24 / OpenDrive
   \ "opendrive"
25 / OpenStack Swift (Rackspace Cloud Files, Memset Memstore, OVH)
   \ "swift"
26 / Pcloud
   \ "pcloud"
27 / Put.io
   \ "putio"
28 / QingCloud Object Storage
   \ "qingstor"
29 / SSH/SFTP Connection
   \ "sftp"
30 / Sugarsync
   \ "sugarsync"
31 / Tardigrade Decentralized Cloud Storage
   \ "tardigrade"
32 / Transparently chunk/split large files
   \ "chunker"
33 / Union merges the contents of several upstream fs
   \ "union"
34 / Webdav
   \ "webdav"
35 / Yandex Disk
   \ "yandex"
36 / http Connection
   \ "http"
37 / premiumize.me
   \ "premiumizeme"
38 / seafile
   \ "seafile"
Storage> box
** See help for box backend at: https://rclone.org/box/ **
```

- Leave blank for the following questions about: client\_id, client\_secret, box\_config\_file, access\_token.

```
OAuth Client Id
Leave blank normally.
Enter a string value. Press Enter for the default ("").
client_id>
OAuth Client Secret
Leave blank normally.
Enter a string value. Press Enter for the default ("").
client_secret>
Box App config.json location
Leave blank normally.

Leading `~` will be expanded in the file name as will environment variables such as `${RCLONE_CONFIG_DIR}`.

Enter a string value. Press Enter for the default ("").
box_config_file>
Box App Primary Access Token
Leave blank normally.
Enter a string value. Press Enter for the default ("").
access_token>
```

- Type user for the option to delegate the connection role to rclone.

```
Enter a string value. Press Enter for the default ("user").
Choose a number from below, or type in your own value
 1 / Rclone should act on behalf of a user
   \ "user"
 2 / Rclone should act on behalf of a service account
   \ "enterprise"
box_sub_type> user
```

- Use the default values for the rest of the questions for: Edit advanced config? Use auto config? Then, It will provide you a link and wait for code.

```
Edit advanced config? (y/n)
y) Yes
n) No (default)
y/n>
Remote config
Use auto config?
 * Say Y if not sure
 * Say N if you are working on a remote or headless machine
y) Yes (default)
n) No
y/n>
If your browser doesn't open automatically go to the following link: http://127.0.0.1:53682/auth?state=#####################
Log in and authorize rclone for access
Waiting for code...
```

- Open your browser to the link on your machine where rclone config has been running on.
- Login to WashU Box with your credential. Approve the access on your Duo App.

![](../../attachments/f658a21a-7eab-4ed3-8be5-9b09775fc099.png)

- Grant the access for rclone to connect to Box. Then, your will see the confirmation of the process. An email notification from box will be sent to you with the subject: Box login from “rclone”.

![](../../attachments/7cd0aa36-85ef-42a2-aaa9-f43ab3295445.png)![](../../attachments/ca025938-41b9-4af9-8c70-00ea64255416.png)

- Close the browser. The configuration for rclone connection to Box will be displayed on your terminal. For example:

```
Got code
--------------------
[Box]
type = box
box_sub_type = user
token = {"access_token":"###########################","token_type":"bearer","refresh_token":"##############################################","expiry":"2020-12-11T12:45:22.744758-06:00"}
--------------------
y) Yes this is OK (default)
e) Edit this remote
d) Delete this remote
y/e/d>
```

- Type y if the configuration content looks OK. Then, you will see the new remote connection in the remotes list.

```
y/e/d> y
Current remotes:

Name                 Type
====                 ====
Box                  box
```

- Type q to finish the interactive process.

```
e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
e/n/d/r/c/s/q> q
```

### Copying the credential file to the home directory on compute1

#### Confirm the rclone configuration file from the terminal where rclone config has been run.

- On Mac and Linux:

```
> ls -la $HOME/.config/rclone/rclone.conf
```

- On Windows (using CMD or PowerShell):

```
> dir %APPDATA%/rclone/rclone.conf
```

> [!IMPORTANT]
> Windows Command Assumptions:
>
> The above command assumes the `rclone` configuration file is its default folder. Please see the [rclone documentation](https://rclone.org/docs/#config-config-file) for more information.
>
> It is also assumed that the `%APPDATA%` environment variable is set to the correct location. Replace `%APPDATA%` with the correct path if needed.

#### (Optional) Verify the content of the file to see the remote storage you’ve just created.

- On Mac and Linux:

```
> view $HOME/.config/rclone/rclone.conf
```

- On Windows (using CMD or PowerShell):

```
> type %APPDATA%/rclone/rclone.conf
```

#### Copy the file to your compute1 home directory. For example (replacing `<washukey>` with your WashU key):

- On Mac and Linux:

```
> scp $HOME/.config/rclone/rclone.conf <washukey>@compute1-client-1.ris.wustl.edu:~/.rclone.conf
```

- On Windows (using CMD or PowerShell):

```
> scp %APPDATA%/rclone/rclone.conf <washukey>@compute1-client-1.ris.wustl.edu:~/.rclone.conf
```

## Test

- Run ssh to a compute1 client from a terminal. You will get a shell at your compute1 home.
- Verify the rclone configuration file at your home directory.

```
> ls -la .rclone.conf
```

- Run bsub to start a rclone container on a compute1 exec node.

```
> LSF_DOCKER_ENTRYPOINT=/bin/sh bsub -Is -G group-name -q general-interactive -a 'docker(rclone/rclone)' /bin/sh
```

- Run rclone lsd to check the connection from compute1 exec node to your Box storage by listing the directories. For example:

```
> rclone lsd Box:/
```

# Use Case

## From Box to Storage1

### Example: A user has a file File\_A in the WashU Box. The file needs to be moved to the storageN space /storageN/fs1/${STORAGE\_ALLOCATION}/Active.

- Run ssh to a compute1 client from a terminal. For example:

```
> ssh compute1-client-1.ris.wustl.edu
```

- Verify the rclone configuration file is in the home directory.

```
> ls -la $HOME/.rclone.conf
```

- Prepare to mount the storageN space to the job.

```
> export LSF_DOCKER_VOLUMES=/storageN/fs1/${STORAGE_ALLOCATION}/Active:/storageN/fs1/${STORAGE_ALLOCATION}/Active
```

- Run bsub to start a rclone container.

```
> LSF_DOCKER_ENTRYPOINT=/bin/sh bsub -Is -G group-name -q general-interactive -a 'docker(rclone/rclone)' /bin/sh
```

- Copy File\_A from the WashU Box to the storageN space.

```
> rclone ls Box:/File_A
314572800 File_A

> ls /my_storageN/File_A
ls: /my_storageN/File_A: No such file or directory

> rclone copy Box:/File_A /my_storageN/
```

- Verify the file in the storageN space.

```
> ls /my_storageN/File_A
/my_storageN/File_A
```

- Exit the rclone container.

```
> exit
```
