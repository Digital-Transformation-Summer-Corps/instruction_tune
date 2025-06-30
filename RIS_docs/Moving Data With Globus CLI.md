
[Storage Platforms](../Storage%20Platforms.md)

# Moving Data With Globus CLI

- [Installing Globus](#installing-globus)
- [Login Into Globus CLI](#login-into-globus-cli)
- [Globus CLI options](#globus-cli-options)
- [Viewing Transfer and Task Information](#viewing-transfer-and-task-information)
- [Further Globus Options](#further-globus-options)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Installing Globus

- You can install the Globus CLI by following the instructions [found here.](https://docs.globus.org/cli/installation/)
- Or you can use the RIS developed Dockerimage for Globus CLI. [Globus CLI](../Applications/Compute1%20Applications/RIS%20Developed%20Containers/Globus%20CLI.md)

# Login Into Globus CLI

- Once you have installed the Globus CLI or are using the Dockerimage, you will need to login. When using the dockerimage you will need to include the `--no-local-server` option and follow the directions it gives.

```
globus login

globus login --no-local-server
```

# Globus CLI options

- Below is a list of some of the options available when using the Globus CLI.

## Globus `endpoint` Command

- The `endpoint` command has many options, but the one that is most useful is the search option as you can use this option to search for the ID of an endpoint via name.

```
globus endpoint search "WashU RIS"
```

- An endpoint search using WashU as the search terms with bring up any endpoints that are related to that term. Multiple endpoints will show up, but the one you want to use is RIS Storage1.

```
ID                                   | Owner                                                        | Display Name
------------------------------------ | ------------------------------------------------------------ | -----------------------------------------
b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4 | 6f79e659-fa99-4fdc-aa56-1d7076296b72@clients.auth.globus.org | RIS Storage1
```

- The ID is the information that will be utilized in the other options. Please note that the correct RIS endpoint is the one named RIS Storage1. The others are being phased out.

## Globus `ls` Command

- Once you have the ID of the endpoint you wish to use, you can use `globus ls` just like you use ls in linux/unix to display the files and directories in a directory.
- The RIS storage1 collection is the storage endpoint and will display what is in your storageN directory. The following command demonstrates how to do that.

```
globus ls b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/storageN/fs1/washukey/Active/
```

## Globus `transfer` Command

- Once you have determined the endpoints that you wish to use to transfer data, you can utilize the transfer command. You can transfer between the WashU endpoints to move data that way.
- There are three ways you can transfer data.

  - Transfer individual files.
  - Transfer a directory, using the recursive `-r` option.
  - Transfer files in bulk using the `--batch` option.
- Transfering an invdividual file.

```
globus transfer b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/source/location/my_file.txt b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/destination/location/my_file.txt
```

- Transfering a directory.

```
globus transfer -r b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/source/location/my_directory b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/destination/location/my_directory
```

- Transfering using the `--batch` option

  - You need to provide the directories that you are transferring between like with the transfer of a directory.
  - You need a space separated text file with the location of the files in the source endpoint followed by the location in the destination endpoint. Each file must have it’s own line.

  ```
  Sample1.txt Group1/Sample1.txt
  Sample2.txt Group1/Sample2.txt
  Sample3.txt Group1/Sample3.txt
  Sample4.txt Group2/Sample4.txt
  Sample5.txt Group2/Sample5.txt
  ```

  - With the batch method, you can sort files from the source into separate directories in the destination or pull files from separate directories in the source into one directory in the destination.

  ```
  Group1/Sample1.txt Sample1.txt
  Group1/Sample2.txt Sample2.txt
  Group2/Sample3.txt Sample3.txt
  Group2/Sample4.txt Sample4.txt
  Group2/Sample5.txt Sample5.txt
  ```

  - You can provide the file of the files you wish to transfer to the command with the `<` option.

  ```
  globus transfer --batch b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/source/location/my_directory b9545fe1-f647-40bf-9eaf-e66d2d1aaeb4:/path/to/destination/location/my_directory < files_to_transfer.txt
  ```

# Viewing Transfer and Task Information

> [!IMPORTANT]
> Globus retains task information for 90 days. This is not controlled by RIS.

> [!IMPORTANT]
> Commands in this section support the `-F/--format json` option for easier programmatic processing.

## Transfer Status

Given a transfer task ID, information about the status of the task can be retrieved with `task show`:

```
[user@host cwd]$ globus task show 543e219c-01c1-11eb-81a3-0e2f230cc907
Label:                   None
Task ID:                 543e219c-01c1-11eb-81a3-0e2f230cc907
Is Paused:               False
Type:                    TRANSFER
Directories:             1
Files:                   3
Status:                  SUCCEEDED
Request Time:            2020-09-28T19:32:22+00:00
Faults:                  0
Total Subtasks:          6
Subtasks Succeeded:      6
Subtasks Pending:        0
Subtasks Retrying:       0
Subtasks Failed:         0
Subtasks Canceled:       0
Subtasks Expired:        0
Completion Time:         2020-09-28T19:32:27+00:00
Source Endpoint:         Globus Tutorial Endpoint 1
Source Endpoint ID:      ddb59aef-6d04-11e5-ba46-22000b92c6ec
Destination Endpoint:    RIS Storage1
Destination Endpoint ID: 7e92c283-7513-4d54-acdf-516f027bfeb2
Bytes Transferred:       14
Bytes Per Second:        3
```

## Transferred Filenames

The `task info` command can also produce a list of successfully transferred files when the `-t/--successful-transfers` option is used:

```
[user@host cwd]$ globus task show -t 543e219c-01c1-11eb-81a3-0e2f230cc907
Source Path         | Destination Path
------------------- | --------------------------------------------------------
/~/godata/file1.txt | /storage1/fs1/corcoran.william.p/Active/godata/file1.txt
/~/godata/file2.txt | /storage1/fs1/corcoran.william.p/Active/godata/file2.txt
/~/godata/file3.txt | /storage1/fs1/corcoran.william.p/Active/godata/file3.txt
```

> [!IMPORTANT]
> - This information is only available to the task owner (user), not RIS
> - This information is only available through the CLI and SDK, not the web UI

## Event and Failure Logs

Timing and activity information (including errors) can be obtained with `task event-list`:

```
[user@host cwd]$ globus task event-list 543e219c-01c1-11eb-81a3-0e2f230cc907
Time                      | Code      | Is Error | Details
------------------------- | --------- | -------- | ---------------------------------------------------------------------------
2020-09-28T19:32:27+00:00 | SUCCEEDED |        0 | {"files_succeeded":3}
2020-09-28T19:32:27+00:00 | PROGRESS  |        0 | {"bytes_transferred":14,"duration":1.44,"mbps":0.0}
2020-09-28T19:32:25+00:00 | STARTED   |        0 | {"concurrency":2,"parallelism":4,"pipelining":20,"type":"GridFTP Transfer"}
2020-09-28T19:32:24+00:00 | SUCCEEDED |        0 | n/a
2020-09-28T19:32:24+00:00 | STARTED   |        0 | Starting sync delete
```

# Further Globus Options

- If you would like to learn more about the other options in the Globus CLI, you can find that [information here.](https://docs.globus.org/cli/reference/)
