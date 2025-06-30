
[Storage Platforms](../Storage%20Platforms.md)

# Connecting to Storage from Linux

- [Using mount](#using-mount)
- [Using smbclient](#using-smbclient)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Using `mount`

This was tested on Ubuntu 16.04 with mount.cifs version 6.4 which is part of the `cifs.utils` package

Install `cifs.utils`:

```
sudo apt-get install cifs.utils
```

Use the `id` command do find the local user’s `uid` and `gid`, optional:

```
id

uid=1000({local user name}) gid=1000({local user group}) groups=1000({local user group}),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),110(lpadmin),111(sambashare),119(libvirt)
```

Issue the `mount` command with proper options and credentials:

```
mount -t cifs -o vers=3.0,uid=1000,gid=1000,credentials=/tmp/.auth //storageN.ris.wustl.edu/${VOLUME_NAME} /mnt
```

If using a credentials file, be sure to format as follows, removing any line whitespace:

```
username=<WASHU KEY ID>
password=<WASHU KEY PASSWORD>
domain=ACCOUNTS.AD.WUSTL.EDU
```

> [!IMPORTANT]
> Variable replacement:
>
> - ${VOLUME\_NAME} should be replaced with the name of your Storage Platform volume.
> - <WASHU KEY ID> should be replaced with your WashU key ID.
> - <WASHU KEY PASSWORD> should be replaced with the password associated with your WashU key.

# Using smbclient

## Install `samba-client`

In the Debian family of operating systems (e.g. Debian, Ubuntu, etc.)

```
sudo apt-get update && sudo apt-get install -y samba-client
```

In the RedHat family of operating systems (e.g. RedHat, CentOS, etc.)

```
sudo yum install -y samba-client
```

## Empty the default samba configuration

```
sudo truncate -s 0 /etc/samba/smb.conf

Record your credentials in a file with restricted permissions
```

```
cat > secret <<EOF
username=<WASHU KEY ID>
password=<WASHU KEY PASSWORD>
domain=ACCOUNTS.AD.WUSTL.EDU
EOF
```

```
chmod 0600 secret
```

> [!IMPORTANT]
> Variable replacement:
>
> - <WASHU KEY ID> should be replaced with your WashU key ID.
> - <WASHU KEY PASSWORD> should be replaced with the password associated with your WashU key.

## Use `smbclient` to connect to storage with the SMB3 protocol to your storage volume.

```
smbclient -A secret -m SMB3 //storageN.ris.wustl.edu/${VOLUME_NAME} -c ls

Domain=[ACCOUNTS] OS=[] Server=[]
  .                                   D        0  Wed Jan 31 10:35:05 2018
  ..                                  D        0  Wed Jan 31 12:40:03 2018
  Active                              D        0  Wed Jan 31 12:09:39 2018
  .snapshots                        DHR        0  Mon Jan 29 15:22:55 2018

      5368709120 blocks of size 1024. 5363531776 blocks available
```

Or, to avoid recording credentials in a file:

```
smbclient -W ACCOUNTS -U <WASHU KEY ID> -m SMB3 //storageN.ris.wustl.edu/${VOLUME_NAME} -c 'ls'
```

> [!IMPORTANT]
> Variable replacement:
>
> - ${VOLUME\_NAME} should be replaced with the name of your Storage Platform volumn.
> - <WASHU KEY ID> should be replaced with your WashU key ID.

Enter `ACCOUNTS\$USER` password

```
Domain=[ACCOUNTS] OS=[] Server=[]
  .                                   D        0  Wed Jan 31 10:35:05 2018
  ..                                  D        0  Wed Jan 31 12:40:03 2018
  Active                              D        0  Wed Jan 31 12:09:39 2018
  .snapshots                        DHR        0  Mon Jan 29 15:22:55 2018

      5368709120 blocks of size 1024. 5363531776 blocks available
```

## Use `smbclient` to PUT data into storage

```
smbclient -A secret -m SMB3 //storageN.ris.wustl.edu/${VOLUME_NAME} -c 'cd Active; put somefile;'

Domain=[ACCOUNTS] OS=[] Server=[]
putting file somefile as \Active\somefile (69719.1 kb/s) (average 69719.1 kb/s)
```

> [!IMPORTANT]
> Variable replacement:
>
> - ${VOLUME\_NAME} should be replaced with the name of your Storage Platform volumn.
