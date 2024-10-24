# Use a smaller CUDA runtime image instead of the devel version
FROM tensorpod/torch-and-tf2-on-cuda:pt2.3-cu118-tf2.14.0-py3.9-no-ta-tv

LABEL maintainer="Amadou Wolfgang Cisse <amadou.6e@googlemail.com>"

ARG DEBIAN_FRONTEND=noninteractive
ENV IS_DOCKER=1
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install minimal dependencies
RUN apt-get update -q && \    
    apt-get install -y --no-install-recommends make \
    sudo \
    curl \
    gpg \
    gnupg \
    libssl-dev \
    build-essential \
    apt-transport-https \
    lsb-release \
    jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /workdir
COPY . .
RUN rm -rf .github && chmod -R +x /workdir

# Install the conda environment
RUN conda env update -n base -f ./docker/dependencies/linux-gnu-image.yml && \
    /opt/conda/bin/conda clean -afy

# Clean up any unnecessary conda and pip files
RUN find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    rm -rf /root/.cache/pip 

# Docker GPG keys
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update -y && apt-get install -y docker-ce-cli

# Install MongoDB
RUN wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add - && \
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list && \
    apt-get update -y && apt-get install -y mongodb-org && \
    mkdir -p /workdir/data/db 
# && \
# echo 'mongod --dbpath /workdir/data/db --fork --logpath /var/log/mongod.log --bind_ip_all' >> /root/.bashrc

RUN echo "source activate base" >> ~/.bashrc && \
    echo "make" >> ~/.bashrc && \
    echo "source etc/config_env.sh" >> ~/.bashrc


# Set Docker host for TCP connection
ENV DOCKER_HOST=tcp://host.docker.internal:2375

CMD ["/bin/bash"]
