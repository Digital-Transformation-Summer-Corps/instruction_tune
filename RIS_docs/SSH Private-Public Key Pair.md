


# SSH Private-Public Key Pair

SSH keys can be used to authenticate to the cluster. This is the recommended method and allows you to securely SSH to the compute client without entering a passphrase.

1. Generate key files `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub` on your Linux/Unix/MacOS system, using the command `ssh-keygen`:

```
> ssh-keygen -f ~/.ssh/id_rsa_compute
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /Users/${compute_username}/.ssh/id_rsa_compute.
Your public key has been saved in /Users/${compute_username}/.ssh/id_rsa_compute.pub.
The key fingerprint is:
SHA256:P10QiInCIHPUMeIiiP/wtOQwW1D6eItQ75o3pDn+vKk ${compute_username}@macbook.local
The key's randomart image is:
+---[RSA 2048]----+
|o.=++. . o ..    |
|o= ++.. o .  .   |
|= =  .      .    |
|.+ =         .   |
|. * B   S     .  |
| . /.o   . . .   |
|  o+B     o .    |
|  ++o.     .     |
| .E==o           |
+----[SHA256]-----+
```

To avoid typing the passphrase for your key, use ssh-agent:

```
> eval `ssh-agent`
Agent pid 76698
```

2. Add your key to the agent:

```
> ssh-add ~/.ssh/id_rsa_compute
Enter passphrase for /Users/${compute_username}/.ssh/id_rsa_compute:
```

3. Now copy this SSH ID to the compute client:

```
> ssh-copy-id -i ~/.ssh/id_rsa_compute ${compute_username}@compute1-client-1.ris.wustl.edu
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/Users/${compute_username}/.ssh/id_rsa_compute.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
${compute_username}@compute1-client-1.ris.wustl.edu's password:

Number of key(s) added:        1

Now try logging into the machine, with:   "ssh '${compute_username}@compute1-client-1.ris.wustl.edu'"
and check to make sure that only the key(s) you wanted were added.
```

```
> ssh ${compute_username}@compute1-client-1.ris.wustl.edu
Last login: Mon Oct 28 11:32:02 2019 from 10.23.317.459

> whoami
${compute_username}
```

You are now able to securely SSH to the compute client without entering a passphrase.
