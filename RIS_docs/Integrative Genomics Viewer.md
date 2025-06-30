
[Applications](../../../Applications.md) > [Compute1 Applications](../../Compute1%20Applications.md) > [RIS Developed Containers](../RIS%20Developed%20Containers.md)

# Integrative Genomics Viewer

- [Software Included](#software-included)
- [Build IGV Docker Image](#build-igv-docker-image)
- [Submitting the IGV job](#submitting-the-igv-job)

> [!IMPORTANT]
> Compute Resources
>
> - Have questions or need help with compute, including activation or issues? Follow [this link.](https://washu.atlassian.net/servicedesk/customer/portal/2/group/6/create/43)
> - [RIS Services Policies](../../../RIS%20Services%20Policies.md)

> [!IMPORTANT]
> Docker Usage
>
> - The information on this page assumes that you have a knowledge base of using Docker to create images and push them to a repository for use. If you need to review that information, please see the links below.
> - [Docker and the RIS Compute1 Platform](../../../Compute1/Docker%20and%20the%20RIS%20Compute1%20Platform.md)
> - [Docker Basics: Building, Tagging, & Pushing A Custom Docker Image](../../../Docker/Docker%20Basics_%20Building,%20Tagging,%20&%20Pushing%20A%20Custom%20Docker%20Image.md)

# Software Included

- noVNC (<https://novnc.com/info.html> )
- IGV for Linux (<https://www.igv.org/> )

# Build IGV Docker Image

1. Log onto compute1

```
> ssh washukey@compute1-client-1.ris.wustl.edu
```

2. Create IGV-specific noVNC `Dockerfile` by entering the following command in its entirety:

```
> cat <<'EOF' > $HOME/noVNC/Dockerfile

FROM gcr.io/ris-registry-shared/novnc:ubuntu20.04
# Install IGV
ENV IGV_URL="https://data.broadinstitute.org/igv/projects/downloads/2.9/IGV_Linux_2.9.4_WithJava.zip"
RUN wget ${IGV_URL} -O IGV.zip \
    && IGV_DIR=`zipinfo -1 IGV.zip | cut -f 1 -d '/' | sort | uniq` \
    && unzip IGV.zip \
    && mv $IGV_DIR IGV \
    && rm IGV.zip \
    && chmod -R 777 IGV \
    && cd IGV \
    && chmod a+x igv.sh
ENV PATH=/app/IGV:$PATH
EOF
```

3. Build and push your Docker container to [Docker Hub](https://hub.docker.com/). Documentation on how to do this can be found [Docker Tutorial](../../../Docker/Docker%20Tutorial.md).

# Submitting the IGV job

1. Set the noVNC password

```
> export PASSWORD=
```

> [!WARNING]
> Complete export PASSWORD= with the password of your choosing. Do not leave this blank.

2. Submit the IGV job to compute1

```
> PATH="/app/IGV:$PATH" LSF_DOCKER_PORTS='8080:8080' bsub -Is -R 'select[port8080=1]' -q general-interactive -a 'docker(location-of-docker-image)' supervisord -c /app/supervisord.conf
```

- Please see our documentation for more information on selecting a port.

> [!IMPORTANT]
> You are a member of multiple LSF User Groups:
>
> If you are a member of more than one compute group, you will be prompted to specify an LSF User Group with -G group\_name or by setting the LSB\_SUB\_USER\_GROUP variable.

- Since LSF is running interactively, it will output the name of the host it’s running on in the terminal.
- The host will be the IP address needed to access the VNC.
- For example: `<<Starting on compute1-exec-187.ris.wustl.edu>>` translates to the IP being `https://compute1-exec-187.compute.ris.wustl.edu:8080/vnc.html`
- The password will be what was set above with the `export PASSWORD=` command.

3. Once connected, you can open IGV by entering `igv.sh` into the GUI terminal.

```
> igv.sh
```

![image-20250314-124345.png](../../../../attachments/64e668ba-64a2-40dd-9ded-a203dc4b0a3e.png)
> [!IMPORTANT]
> Using a different IGV version:
>
> The above instructions are current for IGV v.2.9.4. If you would like to use another version of IGV, substitute the IGV\_URL in the Dockerfile with the URL of the version of IGV for Linux you would like to use.
