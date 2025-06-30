
[Compute1](../Compute1.md)

# RTM

- [Overview](#overview)
- [RTM Dashboard](#rtm-dashboard)

# Overview

## What is RTM?

IBM® Spectrum LSF RTM (RTM) is an operational dashboard for IBM Spectrum LSF environments that provides comprehensive workload monitoring, reporting, and management. It makes cluster administrators more efficient in their day-to-day activities and provides the information and tools that are needed to improve cluster efficiency, enable better user productivity, and contain or reduce costs.

## How to Access RTM

You can access RTM for the RIS Compute Platform at the following link.

<https://compute1-rtm-1.ris.wustl.edu/cacti/index.php>

## How to Login

- You will use your washukey and password to login to RTM.
- There are multiple options to choose from on login, most users will want to use the Compute login.

![image-20250314-131740.png](../../attachments/70942af9-939c-41e9-ba3a-99e05785782e.png)

# RTM Dashboard

## Cluster Dashboard

- This dashboard contains information for all the the execution nodes (hosts).
- This is where users can search jobs by user, JobID, queue, processor model and type, and status of the execution node (host).

### Actions

- View Active Jobs: Shows the basic details of job running on the execution node.

![image-20250314-131949.png](../../attachments/e184eaa7-3132-4d2c-978e-706b4954a318.png)![image-20250314-132000.png](../../attachments/e7ece5c6-5993-4fcd-8c34-b02e936dd058.png)

- View Host Job Detail: Shows the details of the execution node (host) itself.

![image-20250314-132122.png](../../attachments/a140dfdb-e32d-4551-8747-03e8f2920b03.png)![image-20250314-132131.png](../../attachments/ba8477d8-d225-426d-b325-4e338ccb3aa7.png)

- View Host Graphs: Shows graphs of usage of the execution node (host).

![image-20250314-132221.png](../../attachments/a1b137cd-c4f6-4485-b63b-4399618ad526.png)![image-20250314-132230.png](../../attachments/7fc48932-29d5-4056-b5d0-efeab47febca.png)

- Details

  - **Actions:** Detailed above.
  - **Host Name:** The LSF execution node (host) name.
  - **Load/Batch:** The status of the execution node (host).
  - **CPU %:** The execution nodes’s (host’s) overall CPU utilization rate, as a percentage.
  - **RunQ 1m:** The exponentially-averaged effective CPU run queue length for this execution node (host) over the last minute.
  - **Mem Usage:** The execution node’s (host’s) memory usage, as a percentage.
  - **Page Usage:** Page usage of all jobs running on this execution node (host).
  - **Page Rate:** Page usage of all jobs running on this execution node (host) as a percentage of total page size.
  - **Max Slots:** The maximum number of job slots on the execution node (host).
  - **Num Slots:** The number of job slots in use.
  - **Run Slots:** The number of job slots with a running job.
  - **SSUSP Slots:** The number of job slots with system suspended jobs.
  - **USUSP Slots:** The number of job slots with user suspended jobs.
  - **Reserve Slots:** The number of job slots that have been reserved.

![image-20250314-132448.png](../../attachments/62886137-7d06-41c8-b031-b62b9196f68a.png)

## JobIQ Dashboard

- This dashboard contains information about your current and recent jobs.
- This is an alternative place to monitor job progess beyond using bjobs.
- The lefthand side lists the detailed information for all current and recent jobs.

  - Current Status by Cluster for My User: This shows a summary of all jobs running on the compute platform.
  - Current Status by Queue/Project for My User: This shows a summary of all jobs running by queue.
  - Daily Throughput by Cluster for My User: This gives a summary of all jobs that have ran or are queued/running for the day.
  - Feature Checkouts for My User:
  - Pending Reasons by Queue for My User: This gives a summary of all jobs that have the Pending status and the reason for the status.
  - Exit Analysis Since ‘XX-XX XX:XX’ by Queue/Project for My User: This gives a summary of all exited jobs since ‘XX-XX XX:XX’.
- The righthand side shows graphs summaries of some of the information on the lefthand side.

![image-20250314-132621.png](../../attachments/b6513636-9bcb-4dde-b756-1364249181c6.png)

## Graphs Dashboard

- This dashboard contains graphs of the summary of different statistics related to the compute platform as a whole.
- If you click the ‘Cluster - compute1-lsf’, you will see the graphs.
- At the top of the page in Graph Filters, there are multiple options to search and adjust the graphs.

![image-20250314-132802.png](../../attachments/76d2cc72-e0b1-4764-a3be-de81f3c5cc68.png)

## License Dashboard

- This dashboard contains the licenses that are available for use on the compute platform.
- Some users have private licenses stored in other methods, but this contains the RIS supported licenses available.

![image-20250314-132840.png](../../attachments/b5d60d16-c29d-4a59-a0cc-8266fd0e5121.png)

### Actions

- View License Inventory: This shows all the features included in the license.

![image-20250314-132933.png](../../attachments/a58afe4a-c2a2-45e3-ad0c-2dc6d0bbf24f.png)![image-20250314-132944.png](../../attachments/bcc6d6b3-c7b1-48ae-835e-997c2a0ea240.png)

- View License Usage: This shows which feature licenses are in use and how many of the total licenses are used.

![image-20250314-133000.png](../../attachments/5fbb9236-cd57-4190-ae86-847925b1fa5e.png)![image-20250314-133007.png](../../attachments/49c3ddd7-9262-4a6e-97ff-fb118000cddc.png)

- View Reports: This is the main license page you land on that shows a summary of all available licenses.

![image-20250314-133019.png](../../attachments/d68462e4-bb5a-4bd4-9f70-3d7aa9647286.png)![image-20250314-133028.png](../../attachments/4dd2a39b-9b5e-486b-bd36-277c36c8a792.png)

- View License Checkouts: Comprehensive list of the checkouts of the license including user, duration, and date.

![image-20250314-133042.png](../../attachments/e7167be2-c50a-4584-8514-86d44cd63566.png)![image-20250314-133050.png](../../attachments/65631edd-9b17-4247-9baf-7c1cf7d1cbda.png)
