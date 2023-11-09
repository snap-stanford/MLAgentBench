FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

WORKDIR /app
USER root

# Add the current directory contents into the container at /app
COPY install.sh .
COPY requirements.txt .
RUN chown user:user install.sh
RUN apt-get update && apt-get install -y --no-install-recommends gcc &&  rm -r /var/lib/apt/lists/*
RUN apt-get update &&  apt-get install unzip
USER user

# Install libraries 

RUN conda create -n mlagentbench python=3.10
# Make RUN commands use the new environment:
RUN conda init bash
SHELL ["conda", "run", "-n", "mlagentbench", "/bin/bash", "-c"]

RUN echo "conda init bash" > ~/.bashrc
RUN echo "source activate mlagentbench" > ~/.bashrc
RUN /bin/bash -c 'chmod +x install.sh'

RUN /bin/bash -c ./install.sh
ENV PATH /opt/conda/envs/envname/bin:$PATH

