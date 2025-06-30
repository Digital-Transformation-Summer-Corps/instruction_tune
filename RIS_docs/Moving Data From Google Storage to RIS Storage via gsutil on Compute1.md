
[Storage Platforms](../Storage%20Platforms.md)

# Moving Data From Google Storage to RIS Storage via gsutil on Compute1

- [What is this Documentation?](#what-is-this-documentation)
- [Quick Start](#quick-start)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# What is this Documentation?

This documentation will cover doing file transfers with gsutil over our dedicated fiber interconnect in order to download data from sources that use Google storage.

# Quick Start

## 1. Login to the Compute platform

```
ssh washukey@compute1-client-1.ris.wustl.edu
```

## 2. Set up Google Account variable

```
export GOOGLE_ACCOUNT=washukey@wustl.edu
```

> [!IMPORTANT]
> Account Information:
>
> - This is the account that has been granted access to the data by the data owner.
> - This is not necessarily just an email address.

## 3. Login with gcloud

```
gcloud auth login $GOOGLE_ACCOUNT
```

- Follow the URL that you are given.

![image-20250317-153003.png](../../attachments/8fea220a-abb3-41f5-92bd-4a28ad0d9ed3.png)

- It will ask your to sign into Google Cloud SDK. Use the email granted permission by the data owner here, e.g. washukey@wustl.edu.

![image-20250317-153036.png](../../attachments/04b309d4-2e53-4a15-9708-24c5a6c1093a.png)

- This will take you to a WashU authentication page if it’s your WashU email. Put in your information for the WashU sign in you normally would.

![image-20250317-153103.png](../../attachments/de1c7bac-aa50-459e-ab69-04b83f1165c4.png)

- It will then ask for access to your account, click Allow.

![image-20250317-153123.png](../../attachments/55dba1ec-f066-422e-8075-91ace9e1428e.png)

- It will give you a code, you will need to paste this code back into the terminal you are working in.

![image-20250317-153154.png](../../attachments/60f2b1e9-db2c-4514-b351-5e0df7ffc75f.png)![image-20250317-153217.png](../../attachments/3f35233d-3edf-4c36-8bd7-aa16dc61f6c5.png)

- You can confirm this worked with the following command:

```
gcloud auth list
```

![image-20250317-153252.png](../../attachments/3df70d3c-7e00-427a-a1f9-25f8f30fadef.png)
> [!IMPORTANT]
> Multiple Accounts:
>
> - If this account is the only account listed, it will by default be the “active” one.
> - If there are multiple accounts, you can use the following to set the one being used active.
>
> ```
> gcloud config set account $GOOGLE_ACCOUNT
> ```

## 4. Transferring the Data

- Set the following variables needed for the transfer.

```
export GOOGLE_STORAGE_PATH=gs://path/from/data/provider
export DESTINATION_DIR=/storageN/fs1/${STORAGE_ALLOCATION}/Active/path/to/directory
```

> [!IMPORTANT]
> //path/from/data/provider:
>
> //path/from/data/provider is the location of the data on Google Storage, as provided by the group sharing the data.

- You will need to set the following variables

```
export LSB_JOB_REPORT_MAIL=N
export LSF_DOCKER_VOLUMES="/storageN/fs1/${STORAGE_ALLOCATION}/Active/path/to/directory:/data"
export LSF_DOCKER_ADD_HOST=storage.googleapis.com:199.36.153.4
```

- You need to launch a bsub job to use google cloud tools

```
bsub -Is -q general-interactive -a 'docker(google/cloud-sdk)' /bin/bash
```

> [!IMPORTANT]
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with -G group\_name or by setting the LSB\_SUB\_USER\_GROUP variable.

- The following command will run a trial of the transfer to make sure it works.

```
gsutil rsync -r -n $GOOGLE_STORAGE_PATH /data/
```

- Once the test is complete and there are no issues, run the transfer by removing the -n option.

```
gsutil rsync -r $GOOGLE_STORAGE_PATH /data/
```

- If there are problems or the transfer locks up, you can safely restart the transfer without losing progress as it will continue from where it was stopped just like normal rsync.
- If you are transferring a large number of small files, using parallel transfers may work better.
- The following command will run a transfer in parallel.

```
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M -o GSUtil:parallel_thread_count=16 rsync -r $GOOGLE_STORAGE_PATH /data/
```

> [!WARNING]
> Using the parallel option is done so with the knowledge that it can be error prone.
