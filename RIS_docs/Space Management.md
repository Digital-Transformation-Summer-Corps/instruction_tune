
[Compute1](../Compute1.md)

# Space Management

- [Compute Data Transfer Policy](#compute-data-transfer-policy)
- [Storage Platform Integration Points](#storage-platform-integration-points)
- [Checking Storage Usage](#checking-storage-usage)
- [Staging Data](#staging-data)
- [Addtional Notes](#addtional-notes)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Compute Data Transfer Policy

Supported methods to transfer data into and out of the Scientific Compute Platform are:

1. [Moving Data With Globus](../Storage%20Platforms/Moving%20Data%20With%20Globus.md)
2. [Moving Data With Rclone](../Storage%20Platforms/Moving%20Data%20With%20Rclone.md)
3. Submitting a job using a data transfer tool (e.g. `rsync`, `wget`, `curl`, `scp`)

   - Please do not use these data transfer tools directly on the compute1 client nodes. This type of activity can slow down the client node and negatively affect all users connected to the client nodes.
   - If you require assistance submitting jobs using these tools, please open a ticket at our [Service Desk](https://servicedesk.ris.wustl.edu).

# Storage Platform Integration Points

In addition to the Storage Platform SMB Interface to a given Allocation, the Compute Service reveals another three interfaces to the Storage Service. This brings the total number of different types of locations where RIS Storage Space can be consumed to four:

1. Storage Platform Allocations
2. Storage Platform Allocations’ Caches
3. Home directories in the Compute Platform
4. Scratch space directories in the Compute Platform

Each of these types of locations have different methods and policies for managing and inspecting usage. This helps balance the availability of space and performance with the capabilities of the resources that provide them.

# Checking Storage Usage

## Storage Platform Allocations

An accurate report of a Storage Platform Allocation’s space consumption can only be obtained through the Storage Platform SMB Interface. This is a [Storage Platforms](../Storage%20Platforms.md) of the Storage Platform.

### SMB Inteface

The easiest way to do this is to mount the allocation to your desktop, right-click the mounted folder, and select the appropriate menu option for more information.

Alternatively, maximum and available space can be obtained with `smbclient`:

```
$ smbclient -A .smb_creds -k //storageN.ris.wustl.edu/ris -c du

             137438953472 blocks of size 1024. 135136394752 blocks available
Total number of bytes: 14619
```

The “blocks of size 1024” figure is the limit of how much total space can be consumed, and the “blocks available” figure is how much of that limit is not consumed.[^2]

These values can be converted to TiB[^1]

```
$ bc -q
scale=2
137438953472/1024^3
128.00
135136394752/1024^3
125.85
```

Or subtracted from each other to equate space used:

```
(137438953472-135136394752)/1024^3
2.14
```

Telling us that this Allocation is using 2.14 TiB out of a 128 TiB limit.

### Cache Interface (Storage1)

The cache interface can be measured more simply using a shell on a Compute Platform execution node or condo:

```
$ df -Ph /storage1/fs1/ris
Filesystem      Size  Used Avail Use% Mounted on
cache2-fs1      128T  3.3T  125T   3% /storage1/fs1
```

Notice that this usage is over a TiB more than what SMB reported! That’s because it is *a different location*. Rewriting the same 1 TiB file with different data three times would consume 3 TiB on the cache, but ultimately only use 1 TiB on the Storage Service Allocation, where the data gets flushed for “permanent” storage.

Similarly, data written over SMB, the client (login) nodes, or the interactive nodes but not used by a Compute Platform execution node or condo, will not be pulled into the cache, possibly resulting in a cache interface usage that is lower than the underlying Storage Service Allocation’s actual usage.

Again, this discrepancy is a [Storage Platforms](../Storage%20Platforms.md) of the Storage Service.

Because data can be modified from the execution node and condo cache interface, the SMB interface, the client (login) nodes, and the interactive nodes, the possibility of conflicts arises. Conflicts happen when a file is deleted via SMB before the same file has finished writing back to the home fileset from the cache, or if a file is modified at the same path from another source and cache before the cache has time to write the file back to the home fileset. The cache fileset will detect these conflicts and move the data into hidden directories for manual review, in a way to prevent accidental loss of data in flight. The data in these hidden directories counts towards the cache fileset usage.

If you encounter a conflict, contact the [RIS Service Desk](https://servicedesk.ris.wustl.edu/) for assistance.

## Compute Service Home Directories

Every Compute Service User is assigned a limit of 9 GiB of home directory space on the Compute Platform. This space is restricted at the user level, and can only be checked with the appropriate `mmlsquota` command:

```
mmlsquota -u foouser rdcw-fs2:home1
```

For example, the current logged in user’s home directory usage in automatically scaled units can be obtained like so:

```
$ mmlsquota -u $(id -nu) --block-size auto rdcw-fs2:home1
                         Block Limits                                               |     File Limits
Filesystem Fileset    type         blocks      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
cache1-fs1 home1      USR           1.37G         9G        10G          0     none |     5961       0        0        0     none cache1-gpfs-cluster.ris.wustl.edu
```

There is no SMB interface to this space, and `df` reports space for the entire device, which is shared among all home directories.

## Compute Service Scratch Space

High-performance Scratch Space is typically allocated for each lab as it is onboarded to the Compute Service. This space is restricted at the group level, which should represent an eponymous lab. Because it is a shared device like that for home directories, this usage must also be inspected with the appropriate `mmlsquota` command, referencing a group name and group quota on the scratch device:

```
mmlsquota -g compute-foo scratch1-fs1
```

To see the usage of every compute group the current logged in user belongs to, in automatically scaled units, try something like:

```
$ groups | grep -Po 'compute-\S+' | while read COMPUTE_GROUP
> do ls -ld "/scratch1/fs1/${COMPUTE_ALLOCATION}"
> mmlsquota -g "$COMPUTE_GROUP" --block-size auto scratch1-fs1
> done
drwxr-sr-x. 6 root compute-ris 4096 Aug 23 02:43 /scratch1/fs1/ris

Disk quotas for group compute-ris (gid 1208827):
                         Block Limits                                    |     File Limits
Filesystem type         blocks      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
scratch1-fs1 GRP          2.168T         3T         4T          0     none | 33226772       0        0        0     none
drwx--S---. 2 root compute-corcoran.william.p 4096 Aug 23 02:24 /scratch1/fs1/corcoran.william.p

Disk quotas for group compute-corcoran.william.p (gid 1262586):
                         Block Limits                                    |     File Limits
Filesystem type         blocks      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarks
scratch1-fs1 GRP               0        50T        50T          0     none |        3       0        0        0     none
```

# Staging Data

The Compute Service home directories and Scratch Space are not accessible from outside of the Compute Platform. Data should be staged to these locations from a Storage Service Allocation, and computational result or job output data should then be staged back to a Storage Service Allocation.

# Addtional Notes

> [!WARNING]
> Note that your *compute lab group* and your *storage lab group* **are not the same**. That is, the membership of `storage-foo` and `compute-foo` are likely intentionally different, for specific and meaningful reasons.

**[^1 ]:**Binary terabytes, or “tebibytes”, are base 1024 (that is, there are 1024 gibibytes in every tebibyte, and so on). This comes from the interval between SI suffixes on computers historically being represented by ten binary digits, which is 1024 units in decimal. They are commonly labeled as “TB”, although this can lead to a problematic loss of precision when comparing with values calculated using base 1000.

**[^2 ]:**The figures representing limits and available space are not necessarily a *guarantee* that space is available. It is possible for space to be overprovisioned. This happens when the total space available to *all users* is less than the sum of their quotas. Thus, as every user approaches their quota, there is a potential lower effective limit if the *total space for all users* is exhausted first.
