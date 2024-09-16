# FROM ubuntu:20.04
# FROM tensorflow/tensorflow:latest-gpu
FROM continuumio/miniconda3 AS base

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        wget \
        # bzip2 \
        unzip \
        # python3.9 \
        # python3-pip \
        nano \
        software-properties-common \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Anaconda
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh && \
#     bash Anaconda3-2023.03-Linux-x86_64.sh -b && \
#     rm Anaconda3-2023.03-Linux-x86_64.sh
# 
# # Set PATH environment variable to include Anaconda
# ENV PATH /root/anaconda3/bin:$PATH

# Again, test and train scripts should be executable within the container.
RUN pip install update
RUN python3 -m pip install --upgrade pip

# Copy context
RUN mkdir /workdir
WORKDIR /workdir
COPY . /workdir
RUN chmod -R +x /workdir

RUN bash etc/setup.sh

# FROM base AS final <-- this essentially cleans up the build
# Set the entrypoint to bash with the conda environment activated
ENTRYPOINT ["/bin/bash", "-c", "conda activate mimic3 && exec /bin/bash"]
# RUN pip install -r requirements.txt

# Don't add any CMD or ENTRYPOINT!
