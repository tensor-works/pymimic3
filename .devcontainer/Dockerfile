# Step 1: Build stage to create the archive with git
FROM ubuntu:20.04 AS build

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set up workdir
WORKDIR /src

# Copy the Git repo into the build stage
COPY . .

# Create a tarball of the tracked files
RUN git archive --format=tar HEAD > /tmp/workdir-archive.tar

# Use a smaller CUDA runtime image instead of the devel version
FROM tensorpod/torch-and-tf2-on-cuda:pt2.3-cu118-tf2.14.0-py3.9-no-ta-tv

LABEL maintainer="Your Name <your.email@example.com>"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install minimal dependencies
RUN apt-get update -q && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /workdir
# Copy only the files tracked by Git (ignoring .gitignore files)
COPY --from=build /tmp/workdir-archive.tar /workdir

# Extract the tar archive into the workdir
RUN tar -xvf workdir-archive.tar && \
    rm workdir-archive.tar && \
    rm -rf .github

RUN chmod -R +x /workdir

# Activate the environment and install pip packages
RUN conda env update -n base -f .devcontainer/linux-gnu.yml && \
    /opt/conda/bin/conda clean -afy

# Clean up any unnecessary conda and pip files
RUN conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN rm -rf /root/.cache/pip 

# Activate the environment and run the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source activate base && exec /bin/bash"]

