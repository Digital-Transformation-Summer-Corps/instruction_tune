
[Storage Platforms](../Storage%20Platforms.md)

# Troubleshooting Connection to the Storage Platforms

- [Windows troubleshooting](#windows-troubleshooting)
- [Mac and Linux troubleshooting](#mac-and-linux-troubleshooting)

> [!IMPORTANT]
> storageN
>
> - The use of `storageN` within these documents indicates that any storage platform can be used.
> - Current available storage platforms:
>
>   - storage1
>   - storage2

# Windows troubleshooting

Open the Windows Powershell.

![image-20250401-175324.png](../../attachments/a4dab213-3498-4c3b-b216-ff6e1af1bc65.png)

Enter the following command to perform a connection test to the storage service

```
Test-NetConnection -Port 445 -InformationLevel "Detailed" -ComputerName "storageN.ris.wustl.edu"
```

Please be patient as the test may take a few minutes to complete.

### A result of `TcpTestSucceeded : True`indicates that a network connection to the storage service succeeded

![image-20250401-175434.png](../../attachments/b47fe01c-8ebf-4cdf-b0f3-5375215e2140.png)

If you are experiencing problems with accessing your storage allocation on your Windows computer and the above test returned a `TcpTestSucceeded : True` result - please contact our [Service Desk](https://servicedesk.ris.wustl.edu/) for further assistance.

### If the above test fails with a result of `TcpTestSucceeded : False`

![image-20250401-175540.png](../../attachments/4cbfb658-b9df-4a01-b13f-43f7bc50bad6.png)

Please contact [WashU IT Support](https://it.wustl.edu/help/washu-it-support/) or your local network administrator for more help.

# Mac and Linux troubleshooting

Open a terminal and enter the command to perform a connection test to the storage service

```
nc -vz storageN.ris.wustl.edu 445
```

Please be patient as the test may take a few minutes to complete.

### If the test is successful, the following result will appear

![image-20250401-175657.png](../../attachments/09f61032-4484-4aad-8798-39e95525e875.png)

If you are experiencing problems with accessing your storage allocation on your Mac or Linux computer, and the above test was successful - please contact our [Service Desk](https://servicedesk.ris.wustl.edu/) for further assistance.

### A failed test will result in the following

![image-20250401-175722.png](../../attachments/cf8783eb-2c87-43c0-8f98-8f5e1de65f2c.png)

If the test returns a failed result, please contact [WashU IT Support](https://it.wustl.edu/help/washu-it-support/) or your local network administrator for more help.
