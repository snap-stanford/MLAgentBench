FROM mcr.microsoft.com/vscode/devcontainers/anaconda:3

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /app
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc &&  rm -r /var/lib/apt/lists/*
USER user

# Add the current directory contents into the container at /app
COPY install.sh .
COPY requirements.txt .

# Install libraries 

RUN conda create -n mlagentbench python=3.10
# Make RUN commands use the new environment:
RUN conda init bash
SHELL ["conda", "run", "-n", "mlagentbench", "/bin/bash", "-c"]

RUN bash install.sh

RUN echo "conda init bash" > ~/.bashrc
RUN echo "source activate mlagentbench" > ~/.bashrc
ENV PATH /opt/conda/envs/envname/bin:$PATH