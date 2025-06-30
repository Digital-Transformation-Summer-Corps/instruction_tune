
[Research Applications and Services](../Research%20Applications%20and%20Services.md)

# Omero

- [Overview](#overview)
- [Contents](#contents)

# Overview

This page serves as a resource for RIS users of the WashU who wish to leverage [OMERO](https://www.openmicroscopy.org/omero/). OMERO is an Open Microscopy Environment (OME) for labs that are working with CryoEM images in WashU to handle images in a secure central repository. RIS would like to offer the service to the community that use RIS storage service to store their CryoEM data.

# Contents

## Before You Get Started

- If you are reading this document, it is assumed that you are a member of the Washington University user community.
- We assume that you already have an RIS storage allocation. If not please use [this link](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/53) to request one.
- We also assume you have a WashU Key Identity and can access WashU VPN.
- WashU OMERO users agree to abide by the following:

  - [WashU Computer Use Policy](https://informationsecurity.wustl.edu/policies/computer-use-policy)
  - [WashU Information Security Policy](https://informationsecurity.wustl.edu/faculty-staff-students/security-policies-2/information-security-policy)
  - [WashU System Access Control Policy](https://wustl.edu/about/compliance-policies/computers-internet-policies)

## Accessing WashU OMERO

- To access WashU OMERO Portal, please create a ticket for provisioning using the serivce desk portal using this link. Make sure you select “Omero” from the drop down list.
- Then, access WashU OMERO Portal, navigate to the [OMERO page](https://omero.ris.wustl.edu/).

## Setting up OMERO Insight Client

- Please download OMERO Insight client for your respective platform from here.
- Once you launch the OMERO client, you would need to configure it to use the RIS OMERO server.
- Click on the configuration lever button.

![image-20250416-131523.png](../../attachments/0d32650a-3e66-42cb-9a7c-410401d9ca05.png)

- Click on ADD server button

![image-20250416-131544.png](../../attachments/b712cbe6-2a64-42cf-83d4-b170585253f9.png)

- Enter `omero.ris.wustl.edu:4064` in the server address dialog box.

![image-20250416-131610.png](../../attachments/33b48f6c-f244-4a2e-a4ce-097b70b3c16d.png)

- Click on OK button and then click on Apply button.

## Uploading Data Using Insight Client

Once you have logged in to the client. You can upload images using the client. Below we show an example on how to do it:

- Click on File -> New Project Option.
- Provide a name to the Project.

![image-20250416-131700.png](../../attachments/30541941-ae44-4493-885d-f17bdc37849b.png)

- Then Select the Project Name. Click on File -> New Dataset.
- Provide a name for the Dataset.

![image-20250416-131720.png](../../attachments/f0c03e24-03ed-425c-baca-c3734b5fa7ee.png)

- Then double-click on the dataset name from the treeview on the left hand side and click import button.
- You will be presented with an import dialog box wherein on the left hand side pane you will select the file/folder you wish to upload and on the right hand side will be the destination on the omero server.

![image-20250416-131829.png](../../attachments/221df564-996e-4f8b-ab28-90eb9392be73.png)

- Select the file/folder on the left hand pane and click on the right arrow button.
- Select the project and dataset you wish to upload this to.

![image-20250416-131858.png](../../attachments/391b4883-e0c6-4915-aff2-c2e82aac41c6.png)

- Click on add to queue. You can also define tags and other metadata using the options button on the right hand side.

![image-20250416-131916.png](../../attachments/1471beb2-c265-4706-a904-74d8b214c346.png)

- Click on the plus icon to add tags.

![image-20250416-131939.png](../../attachments/c6464bdb-1841-4a34-b33e-ce370763f04a.png)

- Click on Import Button.

![image-20250416-131955.png](../../attachments/149b044d-ec99-4b34-9795-eab46ff2a1f6.png)

## OMERO Support

- RIS does not own the OMERO project for OMERO specific errors we urge users to follow OMERO’s official [support](https://www.openmicroscopy.org/support/).
- If you face issues access the OMERO portal, please raise a ticket through RIS user support [portal](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/53).
