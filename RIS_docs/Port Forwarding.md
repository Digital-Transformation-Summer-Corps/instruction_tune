
[Compute1](../Compute1.md)

# Port Forwarding

- [Video](#video)
- [Start Up GUI Docker](#start-up-gui-docker)
- [Connect With Forwarding](#connect-with-forwarding)
- [Connect Via Browser](#connect-via-browser)

> [!IMPORTANT]
> Compute Resources
>
> - Have questions or need help with compute, including activation or issues? Follow [this link.](https://jira.ris.wustl.edu/servicedesk/customer/portal/1/group/22)
> - [RIS Services Policies](../RIS%20Services%20Policies.md)

# Video

<https://www.youtube.com/watch?v=gGR-9vileCk>

# Start Up GUI Docker

- This works with any docker image that uses noVNC or other software that produces a server-like connection to the GUI through a web browser.
- In this example, we will use the `RIS THPC GUI` to demonstrate.
- Start up your image like you would otherwise.

![image-20250314-131146.png](../../attachments/d1fbd018-4576-40d5-91b8-c1b704911823.png)

# Connect With Forwarding

- Once you have the exec node that the job is running on, you then ssh directly to that node using the following ssh command.

```
ssh -L 8080:compute1-exec-N.compute.ris.wustl.edu:8901 washukey@compute1-client-X.ris.wustl.edu
```

- Where 8080 is the local port you select and 8901 is the port used when running the GUI job, N is the exec node number the job is running on, and X is simply whichever client you choose to use.

![image-20250314-131327.png](../../attachments/9612fbeb-b660-4ae2-a154-d68b94bffb82.png)

# Connect Via Browser

- Once the forwarding is set up, you can connect to the GUI through a web browser like normal, only now you use `https://localhost:8080` where 8080 is the port you chose.

![image-20250314-131407.png](../../attachments/8abc26f1-0b83-45dc-b26d-a23432bf28e4.png)

- Now you have access to your GUI just like you would connecting directly to the exec node via a web browser.

![image-20250314-131434.png](../../attachments/cd26217b-5e60-4b57-9b09-d406703c6f99.png)![image-20250314-131446.png](../../attachments/838a62ae-91d5-4fc9-aaa4-38c589bc4ba1.png)![image-20250314-131507.png](../../attachments/a3316b70-21e3-4932-b5de-4368b44116f2.png)
