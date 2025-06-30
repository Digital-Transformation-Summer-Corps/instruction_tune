


# FAQ

- [What is RIS?](#what-is-ris)
- [How do I contact RIS?](#how-do-i-contact-ris)
- [What are RIS Services?](#what-are-ris-services)
- [Who are RIS Services for?](#who-are-ris-services-for)
- [How do I get started with RIS Services?](#how-do-i-get-started-with-ris-services)
- [How do I get help with RIS Services?](#how-do-i-get-help-with-ris-services)
- [Where can I find RIS Documentation?](#where-can-i-find-ris-documentation)
- [Where can I get help with WashU Keys?](#where-can-i-get-help-with-washu-keys)
- [What do you mean when you say product stages?](#what-do-you-mean-when-you-say-product-stages)
- [Where are the RIS services physically located?](#where-are-the-ris-services-physically-located)
- [How do I Obtain an account?](#how-do-i-obtain-an-account)
- [How do I change my password?](#how-do-i-change-my-password)
- [How do I get an account for a collaborator?](#how-do-i-get-an-account-for-a-collaborator)
- [How do I log into the Compute environment?](#how-do-i-log-into-the-compute-environment)
- [How do I make it so I can log in without having to use my password?](#how-do-i-make-it-so-i-can-log-in-without-having-to-use-my-password)
- [How do I launch jobs in the HPC environment?](#how-do-i-launch-jobs-in-the-hpc-environment)
- [How should I name my files and directories?](#how-should-i-name-my-files-and-directories)
- [What does it mean to have a “Compute Condo(minium)”](#what-does-it-mean-to-have-a-compute-condo-minium)
- [Are there general access computing resources?](#are-there-general-access-computing-resources)
- [What is the difference between the Compute1 general and the general-interactive queues?](#what-is-the-difference-between-the-compute1-general-and-the-general-interactive-queues)
- [What does the Compute Service price include?](#what-does-the-compute-service-price-include)
- [How much space is in my $HOME directory?](#how-much-space-is-in-my-home-directory)
- [Why is this limited to 10G? Can I have more?](#why-is-this-limited-to-10g-can-i-have-more)
- [How do I see what is using up all of my $HOME space?](#how-do-i-see-what-is-using-up-all-of-my-home-space)
- [Why am I getting a Disk I/O error?](#why-am-i-getting-a-disk-i-o-error)
- [How much space is in my Storage Allocation?](#how-much-space-is-in-my-storage-allocation)
- [How do I share files in my storage with colleagues?](#how-do-i-share-files-in-my-storage-with-colleagues)
- [What’s the best way for me to transfer data?](#what-s-the-best-way-for-me-to-transfer-data)
- [How do I request more resources for my job?](#how-do-i-request-more-resources-for-my-job)
- [Does RIS offer Docker containers or a repository for them?](#does-ris-offer-docker-containers-or-a-repository-for-them)
- [Why can’t I connect to my noVNC image?](#why-can-t-i-connect-to-my-novnc-image)
- [Software Debugging Policy](#software-debugging-policy)
- [Troubleshooting your connection to Storage](#troubleshooting-your-connection-to-storage)
- [How do I know what Wash U Network I or my lab is on?](#how-do-i-know-what-wash-u-network-i-or-my-lab-is-on)
- [What are the ways I can access data in the Storage Service?](#what-are-the-ways-i-can-access-data-in-the-storage-service)

> [!IMPORTANT]
> **storageN**
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# What is RIS?

[RIS](https://ris.wustl.edu/) is the Research Infrastructure Services team within Wash U IT. We are a young group, having been incorporated into Wash U IT in 2018. The RIS team strives to build and deliver services related to the research mission of Wash U IT. To date, our service catalog only includes [Research Storage](https://ris.wustl.edu/services/research-storage/) and will soon include [Research Computing](https://ris.wustl.edu/services/compute/). Our product roadmap includes Application Platforms. Generally speaking, RIS attempts to find common patterns in the needs of Wash U research faculty and develops services that addresses the needs these represent. We do not solve all research IT problems, rather we attempt to reduce the costs of the “big ticket items” that represent significant challenges to researchers: high performance and high capacity storage, high performance and high throughput computing, large scale data transfer, and some narrowly focused research applications.

# How do I contact RIS?

RIS uses the [RIS Service Desk](https://servicedesk.ris.wustl.edu/) to interact with users. Each category of the Service Desk includes an “Ask a question” related to that service. This is not the same thing as Service Now, which is seen by some to be a deficiency and by others to be an advantage. We will be working to integrate with Wash U IT and have some linkage between our systems to make it easier for users to contact us.

# What are RIS Services?

RIS services are described in the [RIS Service Catalog](https://wustl.edu/).

# Who are RIS Services for?

RIS services are for Wash U faculty and staff, focused on the research mission of Wash U.

# How do I get started with RIS Services?

It is assumed that users will be research faculty and staff and have WashU Key IDs, have access to Wash U IT Networks, and have access to a departmental WorkDay Cost Center Number. With these as pre-requisites, one can request new services by creating a Service Desk Request via the [RIS Service Desk](https://servicedesk.ris.wustl.edu/).

Below is a guide on how to use the Service Desk to request new RIS services.

- [Requesting RIS Services](Requesting%20RIS%20Services.md)

# How do I get help with RIS Services?

Getting help with RIS services generally falls into a few categories:

- Getting started or enabling services
- Asking questions about services
- Reporting problems with services

Each RIS Service has a section within the [RIS Service Desk](https://servicedesk.ris.wustl.edu/) that contains request forms for these Request Types.

# Where can I find RIS Documentation?

You are reading it! RIS service documentation lives at this Confluence site in two places:

- [The RIS User Documentation](/wiki/spaces/RUD)

# Where can I get help with WashU Keys?

- <https://it.wustl.edu/home/how-to/wustl-connect/>

# What do you mean when you say product stages?

## RIS Product Launch Stages

### **Early Access**

Early-access features are limited to a closed group of testers for a limited subset of launches. Participation is by invitation only. These features may be unstable, change in backward-incompatible ways, and are not guaranteed to be released. There are no SLAs provided and no technical support obligations.

### **Alpha**

Alpha is a limited-availability test before releases are cleared for more widespread use. Our focus with alpha testing is to verify functionality and gather feedback from a limited set of customers. Typically, alpha participation is by invitation and subject to pre-general-availability terms. Alpha releases don’t have to be feature complete, no SLAs are provided, and there are no technical support obligations. However, alphas are generally suitable for use in test environments.

### **Beta**

At beta, products or features are ready for broader customer testing and use. There are no SLAs or technical support obligations in a beta release unless otherwise specified in product terms or the terms of a particular beta program.

### **General Availability**

General availability products and features are open to all customers, covered by the RIS SLAs, and are ready for production use.

### **Deprecated**

Deprecated features are scheduled to be shut down and removed.

Note: Depending on product maturity and engineering needs, a RIS product or feature may not go through every launch stage, and the time between launch phases may vary.

# Where are the RIS services physically located?

- The Research Datacenter is physically located at:

  222 S Newstead Ave.

  St. Louis, MO 63110

  ![](../attachments/6b9079e4-d398-429f-a83c-d966df9c5f0c.png)
- You can take a [virtual tour of the data center](https://kuula.co/share/79Wh6/collection/7lcvK?fs=1&vr=1&sd=1&thumbs=1&chromeless=1&logo=1%22%3E%3C/iframe)

# How do I Obtain an account?

- Accounts for RIS Compute Services use WashU Key credentials, so you must obtain one of those before you can get an account with RIS Compute Services:

  - <https://it.wustl.edu/home/how-to/wustl-connect/>
  - See [this page](https://docs.ris.wustl.edu/doc/01_ris-faq.html#wustl-keys) for help with WashU Keys.

# How do I change my password?

- Since RIS Compute Services uses WashU Key credentials you must change your password for that. Wash U IT has documentation on resetting WashU Key passwords:

  - <https://it.wustl.edu/items/how-do-i-change-my-wustl-key-password/>

# How do I get an account for a collaborator?

- If you are looking to collaborate with someone outside WashU, you will need to have a WashU guest account created for the user.
- Please use this link for starting this process: <https://connect.wustl.edu/guest/guestrequest/>
- Once you have the guest account setup through that process, you can create a ticket in our [Service Desk](https://washu.atlassian.net/servicedesk/customer/portal/2) and we can get the user added to the appropriate allocations.

# How do I log into the Compute environment?

- See our [Compute1 Quickstart](Compute1/Compute1%20Quickstart.md) on how to get connected.

# How do I make it so I can log in without having to use my password?

- We suggest you make use of SSH keys to log into the compute clients, see [SSH Private-Public Key Pair](SSH%20Private-Public%20Key%20Pair.md).

# How do I launch jobs in the HPC environment?

- See our [Compute1 Quickstart](Compute1/Compute1%20Quickstart.md) for submitting your first job. Further information can be found elsewhere in this documentation for more complex examples.

# How should I name my files and directories?

- There is documentation on best practices for file naming available in our documentation. You can find that information at the following links.

  - [RIS Services Policies](RIS%20Services%20Policies.md)
  - [RIS Services Policies](RIS%20Services%20Policies.md)

# What does it mean to have a “Compute Condo(minium)”

- Faculty members may purchase dedicated hardware for their labs to form what we refer to as a “condominium”. In this model, a “condo” is formed out of a set of hardware that we put into a Host Group.
- Then we create a Queue/Partition named after the Lab. E.g. for Compute1: labname and labname-interactive.
- Then we create an AD group named compute-labname and populate it with Users. That group then gets priority access to that lab.

# Are there general access computing resources?

- Yes. The “general” and “interactive” job queues are serviced by a set of execution nodes in a Host Group named “general”.

# What is the difference between the Compute1 general and the general-interactive queues?

- The general queue runs batch jobs much like the traditional HPC setting. They run in the background in the queue system.
- The general queue also makes use of cache system, which you can learn more about [Space Management](Compute1/Space%20Management.md)
- Jobs in the general queue can run for up to 28 days.
- The general-interactive queue runs jobs interactively so that you can interact directly with them or watch a job.
- The general-interactive queue does not use the cache system and instead interfaces with the Storage Platform directly.
- Jobs in the general-interactive queue can run for up to 24 hours.
- Please see the [RIS Services Policies](RIS%20Services%20Policies.md) for more information.

# What does the Compute Service price include?

- Colocation facilities worthy of hosting production quality computing hardware, datacenter space
- Power and cooling of the physical space
- Physical security
- Identity Managment: User accounts, groups, access controls and permissions
- Execution nodes: Varying by CPU flavor, speed, RAM quantity, local hard drive space, etc.
- Networking: All of the above for networking systems
- Storage: All of the above for storage systems
- Data security: Operating system and software updates, incident response
- Integration: Interconnects that provide appropriate bandwidth and Input/Ouput operations per second
- Integration with Cloud Services
- Integration with storage tiers, tape libraries, tape robots, data movers
- Integration with data movement, specialized technologies like Globus
- Operations: Monitoring, alerting, event response
- Support: Help when things go wrong
- Compute job scheduling
- Software development, software artifact repositories
- Container management
- Professional staffing: Specialists in all of the above
- More…

# How much space is in my $HOME directory?

- `$HOME` directories are limited to 10GB. If you wish to observe your quota, you can use the following command:

```
mmlsquota --block-size auto -u washukey rdcw-fs2:home1
```

![image-20250214-165737.png](../attachments/0cae74d6-d92f-4520-80ec-17356a1cc825.png)

- Under the BLock Limits portion ‘blocks’ is how much of the 10Gb that you have consumed.

# Why is this limited to 10G? Can I have more?

- User $HOME directories are intended to allow space for users to make use of the compute platforms, with the knowledge that the Storage Platforms is where data and software will be stored.
- The $HOME directory is required for the Compute Platform(s) to function for users and software often rely on it.
- Policy dictates that you be limited to 10G of $HOME space.
- The $HOME directory is NOT backed up and important data should NOT be stored here. Anything you wish to be backed up should be placed in `/storageN`, this includes scripts.

# How do I see what is using up all of my `$HOME` space?

- You can use the following command to list out the top 10 (or any number if you replace the 10) files or directories using the most space in your $HOME directory.
- Make sure the following command is run from your $HOME directory.

```
du -hsx .[^.]* * 2>/dev/null | sort -rh | head -10
```

- Expected example output.

```
800M        .vscode-server
140M        .local
95M work
68M .cache
41M .lsbatch
24M .nv
21M .matlab
20M .npm
20M .config
15M ondemand

```

# Why am I getting a Disk I/O error?

- This error typically refers to the ability of the job to write a file to a directory.
- The most common source of the error is a user’s home directory being full.
- If you encounter this error, please follow the steps below.

  - Use the methods described in the [home directory space section](https://washu.atlassian.net/wiki/spaces/RUD/pages/edit-v2/1683882153#Why-is-this-limited-to-10G%3F-Can-I-have-more%3F) section to determine if the home directory is at cap.
  - Remove or move files from the home directory to reduce usage.
  - Attempt to run the job again.
  - If the problem persists, submit a ticket to the service desk: <https://ris.wustl.edu/support/service-desk/>

# How much space is in my Storage Allocation?

- The Compute Service is connected to the Storage Service via POSIX filesystem mounts.

  - The batch (execution) nodes and condos are connected via cache.
  - The client and interactive nodes are connected directly.
- The Storage Service provides the SMB interface at `smb://storageN.ris.wustl.edu/${STORAGE_ALLOCATION}`.

  - You can observe available space via SMB mounts with a df command on the mounting workstation.
  - This is for all current storage platforms.
  - This is also the method to use in regards to Storage2 on the Compute Platforms.

```
df --output -h /storage2/fs1/${STORAGE_ALLOCATION}
```

- The Compute Platforms provide a POSIX interface via the filesystem path `/storageN/fs1/${STORAGE_ALLOCATION}`.

  - You can observe available space by the mmlsquota command while logged into the compute platform.
  - This is for the Storage1 Platform.

```
mmlsquota --block-size auto -j washukey_active rdcw-fs1
```

![image-20250214-170156.png](../attachments/f31c570c-48a8-448e-8a39-2a1ea228a391.png)

- Again, under the Block Limits section, the ‘blocks’ portion is how much you have consumed.  
  The Compute Service uses a caching interface to access the data. Read more about how  
  this affects usage and quota here: cache interfaces

# How do I share files in my storage with colleagues?

- You can request access be granted to your colleagues through our ticketing system.

  - <https://jira.wustl.edu/servicedesk/customer/portal/2/group/7>
- You can also use collections within Globus to share specific folders or files with colleagues. This method is the suggested method when it comes to colleagues outside of WashU. You can find more information about using this feature here:

  - [Moving Data With Globus](Storage%20Platforms/Moving%20Data%20With%20Globus.md)

# What’s the best way for me to transfer data?

- The first method we recommend is to use SMB mounts. You can find more information about connecting at the following link.

  - [Storage Platforms](Storage%20Platforms.md)
- Our suggested method of transferring data if SMB is not an option is to make use of Globus. You can use Globus in multiple ways. There are links to our Globus documentation below.

  - [Moving Data With Globus](Storage%20Platforms/Moving%20Data%20With%20Globus.md)
  - [Moving Data With Globus CLI](Storage%20Platforms/Moving%20Data%20With%20Globus%20CLI.md)
  - [Using Globus Connect Personal](Storage%20Platforms/Using%20Globus%20Connect%20Personal.md)

# How do I request more resources for my job?

- Requesting more resources for your job means using options that are part of the bsub command. You can find out more information about the bsub options at the following link.

  - [Job Execution Examples](Compute1/Job%20Execution%20Examples.md)
- Be aware that if the software you use requires special options in order to use these resources, you will need to include those options in your software command as well.

# Does RIS offer Docker containers or a repository for them?

- RIS offers RIS hosted and controlled Docker images.

  - You can find them here: [RIS Developed Containers](Applications/Compute1%20Applications/RIS%20Developed%20Containers.md)
- RIS offers software through our THPC Docker image. Found here: [THPC](Applications/Compute1%20Applications/RIS%20Developed%20Containers/THPC.md)

  - You can find the list of software here: [THPC Installed Applications](Applications/Compute1%20Applications/THPC%20Installed%20Applications.md)
- RIS also offers a list of non-RIS developed containers, where we do not control the Docker image nor host it.

  - You can find that list here: [Non-RIS Docker Images](Applications/Non-RIS%20Docker%20Images.md)
- You can request help building a Docker image if you are having trouble via our ticketing system.
- Software that is used frequently is taken into consideration when creating RIS hosted and controlled Docker images.
- We currently do not have a public repository for users to host their own images in.

# Why can’t I connect to my noVNC image?

- The first reason this could be happening, is port conflicts.

  - If your job lands on a node that has a job already using the port you are attempting to, you will not be able to connect.
  - You can attempt to launch your job on a new node, or you can change the port you’re using and launch the job again.
- The second reason this could be happening, is that some department based VPNs are not part of the trusted network that will allow this.

  - Please see our [Compute1 Quickstart](Compute1/Compute1%20Quickstart.md) for which VPNs we recommend.
- If you wish to avoid dealing with ports for GUI based software, you can check out what software we have available through Open on Demand.

  - [Open OnDemand](Compute1/Open%20OnDemand.md)
- You can also use port forwarding to get around the second reason for being unable to connect.

  - [Port Forwarding](Compute1/Port%20Forwarding.md)

# Software Debugging Policy

We strive to provide help with software debugging and support to the best of our abilities and time. With that being said, there may be times when we cannot solve an issue related to a specific piece of software or script that is not supported by RIS. In those cases, we will attempt to provide a solution to the problem, but we cannot guarantee that the solution will be successful. We recommend reading [Software Development Using Compute1](Compute1/Software%20Development%20Using%20Compute1.md) for more help debugging your software as well as for guidance on software development best practices.

# Troubleshooting your connection to Storage

If you are experiencing issues maintaining a stable connection to your storage allocation, please visit the storage service troubleshooting page.

- [Troubleshooting Connection to the Storage Platforms](Storage%20Platforms/Troubleshooting%20Connection%20to%20the%20Storage%20Platforms.md)

# How do I know what Wash U Network I or my lab is on?

- The RIS team supplies a Speedtest application that will report the IP address of  
  the browsing computer. Visit [the Speedtest URL](https://speedtest.ris.wustl.edu/).

![](../attachments/28849e42-3b44-45d4-97e2-e3190c35c8d6.png)

# What are the ways I can access data in the Storage Service?

At the time of this writing, you can access storage service Allocations via:

- SMB mounts from MacOS, Linux, and Windows.
- Globus Data Transfer endpoints.
- The [Compute1](Compute1.md) and [Compute2](Compute2.md) Platforms.
