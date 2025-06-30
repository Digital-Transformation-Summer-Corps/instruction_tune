
[Storage Platforms](../Storage%20Platforms.md)

# Storage2 Access Control

- [Comparison of Storage1 with Storage2](#comparison-of-storage1-with-storage2)
- [Viewing an ACL on Storage2](#viewing-an-acl-on-storage2)
- [Viewing Share-level Quota Usage on Storage2](#viewing-share-level-quota-usage-on-storage2)
- [Transferring Data Between Storage1 and Storage2 with Globus](#transferring-data-between-storage1-and-storage2-with-globus)

# Comparison of Storage1 with Storage2

The primary difference between accessing Storage1 and Storage2 is their name. For example, a Storage1 allocation may be accessed at via the SMB protocol by mounting `//storage1.ris.wustl.edu/example1`. A similar Storage2 allocation could by accessed by the same protocol by mounting `//storage2.ris.wustl.edu/example2`. From an RIS Compute1 node, the same allocations might be accessed using the Linux VFS interface at `/storage1/fs1/example1` and `/storage2/fs1/example2`.

The second most important difference between the two platforms is the mechanisms for “Discretionary Access Control” (DAC). While Storage1 leverages NFSv4-style **GPFS** ACLs, Storage2 uses NFSv4-compatible ACLs. While they are effectively equivalent in their functionality, they have varying degrees of idiosyncrasy in their usage.

|                 |  Storage1                     |  Storage2                     |
|:----------------|:------------------------------|:------------------------------|
| Interfaces      | SMB, VFS/”POSIX”, NFS, Globus | SMB, VFS/”POSIX”, NFS, Globus |
| Platform        | IBM’s Scale Storage, “GPFS”   | Qumulo Core                   |
| DAC Mechanisms  | NFS4-style GPFS ACLs, “POSIX” | NFS4-compatible ACLs, “POSIX” |
| User Identifier | Unix UID                      | Windows SID                   |

> [!WARNING]
> Be sure to distinguish between **Storage2** and Storage1, **fs2**, which are distinct platforms.

# Viewing an ACL on Storage2

## NFS Protocol

Use the `nfs4_getfacl` command:

```
$ nfs4_getfacl /storage2-dev/fs1/ohids
# file: /storage2-dev/fs1/ohids
A::qumulo_local/admin:rwaDxtTnNcy
A:g:qumulo_local/Users:rwaDxtTnNcy
A:dg:storage-cpohl-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-cpohl-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-cpohl-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-cpohl-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-shin-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-shin-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-shin-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-shin-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-jcaroline-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-jcaroline-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-jcaroline-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-jcaroline-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-testUser-allocation-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-testUser-allocation-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-testUser-allocation-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-testUser-allocation-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-elyn-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-elyn-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-elyn-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-elyn-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-bob_uncle-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-bob_uncle-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-bob_uncle-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-bob_uncle-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-allocation_the_2nd-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-allocation_the_2nd-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-allocation_the_2nd-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-allocation_the_2nd-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-allocation_the_3rd-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-allocation_the_3rd-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-allocation_the_3rd-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-allocation_the_3rd-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-allocation_the_3rd-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-allocation_the_3rd-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-allocation_the_3rd-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-allocation_the_3rd-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-harterj-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-harterj-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-harterj-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-harterj-ro@accounts.ad.wustl.edu:rtncy
A:dg:storage-omero-rw@accounts.ad.wustl.edu:rwaDxtTnNcy
A:fig:storage-omero-rw@accounts.ad.wustl.edu:rwaDtTnNcy
A:dg:storage-omero-ro@accounts.ad.wustl.edu:rxtncy
A:fig:storage-omero-ro@accounts.ad.wustl.edu:rtncy
```

## SMB Protocol

Using the Windows Explorer GUI:

1. Select a file with the cursor and open its context menu (this is usually done with a “right-click” of the pointing device over the respective icon or list item)
2. Select “Properties…”. This should open a new window with an interface to file properties.
3. Click on the “Sharing and Permissions” tab in the file property window.
4. Inspect the displayed ACL.

Using `sbmclient`:

```
$ smbclient //storage2-dev.ris.wustl.edu/Files
Try "help" to get a list of possible commands.
smb: \> showacls
smb: \> ls /storage2-dev/fs1/ohids
FILENAME:ohids
MODE:D
SIZE:0
MTIME:Tue Jul  2 15:32:46 2024
revision: 1
type: 0x8404: SEC_DESC_DACL_PRESENT SEC_DESC_DACL_AUTO_INHERITED SEC_DESC_SELF_RELATIVE
DACL
     ACL     Num ACEs:       46      revision:       2
     ---
     ACE
             type: ACCESS ALLOWED (0) flags: 0x00
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3742419866-1697586437-2279049763-500

     ACE
             type: ACCESS ALLOWED (0) flags: 0x00
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3742419866-1697586437-2279049763-513

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625061

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625061

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625062

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625062

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1982696

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1982696

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1982697

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1982697

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1980602

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1980602

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1980603

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1980603

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952492

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952492

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952493

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952493

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952523

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952523

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952524

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1952524

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625065

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625065

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625066

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625066

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1873315

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1873315

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1873316

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1873316

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625067

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625067

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625068

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625068

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625067

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625067

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625068

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-1625068

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2023215

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2023215

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2023216

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2023216

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1ff
             Permissions: 0x1201ff: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2034846

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x1df
             Permissions: 0x1201df: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2034846

     ACE
             type: ACCESS ALLOWED (0) flags: 0x12  SEC_ACE_FLAG_CONTAINER_INHERIT SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0xa9
             Permissions: 0x1200a9: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2034847

     ACE
             type: ACCESS ALLOWED (0) flags: 0x19 SEC_ACE_FLAG_OBJECT_INHERIT SEC_ACE_FLAG_INHERIT_ONLY SEC_ACE_FLAG_INHERITED_ACE
             Specific bits: 0x89
             Permissions: 0x120089: SYNCHRONIZE_ACCESS READ_CONTROL_ACCESS
             SID: S-1-5-21-3579272529-3368358661-2280984729-2034847

     Owner SID:      S-1-5-21-3742419866-1697586437-2279049763-500
     Group SID:      S-1-5-21-3742419866-1697586437-2279049763-513

             436731904000 blocks of size 512. 431441472904 blocks available
```

Using `smbcacls`:

```
$ smbcacls //storage2-dev.ris.wustl.edu/Files /storage2-dev/fs1/ohids
REVISION:1
CONTROL:SR|DI|DP
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
OWNER:S-1-5-21-3742419866-1697586437-2279049763-500
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
GROUP:S-1-5-21-3742419866-1697586437-2279049763-513
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3742419866-1697586437-2279049763-500:ALLOWED/0x0/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3742419866-1697586437-2279049763-513:ALLOWED/0x0/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625061:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625061:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625062:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625062:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1982696:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1982696:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1982697:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1982697:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1980602:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1980602:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1980603:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1980603:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952492:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952492:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952493:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952493:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952523:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952523:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952524:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1952524:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625065:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625065:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625066:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625066:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1873315:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1873315:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1873316:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1873316:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625067:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625067:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625068:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625068:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625067:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625067:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625068:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-1625068:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2023215:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2023215:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2023216:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2023216:ALLOWED/OI|IO|I/R
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2034846:ALLOWED/CI|I/0x001201ff
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2034846:ALLOWED/OI|IO|I/0x001201df
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2034847:ALLOWED/CI|I/READ
../../source3/rpc_client/cli_pipe.c:550: RPC fault code HRES code 0x16c9a087 received from host storage2-dev.ris.wustl.edu!
ACL:S-1-5-21-3579272529-3368358661-2280984729-2034847:ALLOWED/OI|IO|I/R
```

# Viewing Share-level Quota Usage on Storage2

## SMB Protocol

Using the Windows Explorer GUI:

1. Select the shared drive for the allocation and open its context menu (this is usually done with a “right-click” of the pointing device over the respective icon or list item)
2. Select “Properties…”. This should open a new window with an interface the share properties.
3. Inspect the used and available space.

Use `smbclient` to get raw values, optionally calculating the desired measurements:

```
$ smbclient //storage2-dev.ris.wustl.edu/ohids
Try "help" to get a list of possible commands.
smb: \> ls
  .snapshot                       DHSRn        0  Tue Jul  2 15:32:46 2024
  .                                   D        0  Tue Jul  2 15:32:46 2024
  ..                                  D        0  Thu Sep  5 11:07:44 2024

             10737418240 blocks of size 512. 10737418232 blocks available
smb: \> ^d
$ bc -q
# KB used
(10737418240-10737418232)*512/1024
4
# TB available
scale=3
(10737418240-8)*512/1024^4
4.999
```

## VFS/”POSIX” Protocol

Use the `df` command with the path to a locally mounted share:

```
$ df --output -h /storage2-dev/fs1/ohids
Filesystem                                   Type Inodes IUsed IFree IUse%  Size  Used Avail Use% File                    Mounted on
storage2-dev.ris.wustl.edu:/storage2-dev-fs1 nfs4   1.3G     1  1.3G    1%  5.0T     0  5.0T   0% /storage2-dev/fs1/ohids /storage2-dev/fs1
$ df --output -h /storage2-dev/fs1
Filesystem                                   Type Inodes IUsed IFree IUse%  Size  Used Avail Use% File              Mounted on
storage2-dev.ris.wustl.edu:/storage2-dev-fs1 nfs4    51G  631M   51G    2%  204T  2.5T  201T   2% /storage2-dev/fs1 /storage2-dev/fs1

```

# Transferring Data Between Storage1 and Storage2 with Globus

Globus is an application that serves to move data into the storage service.

Storage2 allocations simply appear as another path on the existing [RIS Globus Collection](https://app.globus.org/file-manager?destination_id=b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4&destination_path=%2F&origin_id=b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4&origin_path=%2F&two_pane=true). Simply select a Storage1 source and a Storage2 destination to move data from Storage1 to Storage2 using Globus.

For more information, see [Moving Data With Globus](Moving%20Data%20With%20Globus.md) or [Moving Data With Globus CLI](Moving%20Data%20With%20Globus%20CLI.md) for more information on using Globus to transfer data.
