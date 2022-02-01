### TODO: Make sure that the cuda driver's version matches (also below)
### TODO: Make sure you have your .dockerignorefile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

### Use bash as the default shelll
RUN chsh -s /bin/bash
SHELL ["bash", "-c"]

### Install basics
RUN apt-get update && \
    apt-get install -y openssh-server sudo nano screen wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

ARG UNAME=user
ARG UID=1002
RUN useradd -rm -d /home/$UNAME -s /bin/bash -g sudo -u $UID $UNAME
RUN echo "$UNAME ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/$UNAME
WORKDIR /home/$UNAME
COPY .ssh /home/$UNAME/.ssh

### Login
USER $UNAME

### Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    sudo /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    sudo find /opt/conda/ -follow -type f -name '*.a' -delete && \
    sudo find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    sudo /opt/conda/bin/conda clean -afy

ENV PATH /opt/conda/bin:$PATH

### Install Environment
ENV envname discrete_bottleneck
RUN sudo env "PATH=$PATH" conda update conda && \
    sudo chown $UID -R /home/$UNAME && \
    conda create --name $envname python=3.8

### Install a version of pytorch that is compatible with the installed cudatoolkit
RUN conda install -n $envname pytorch=1.8.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
COPY requirements.yaml /tmp/requirements.yaml
RUN conda env update --name $envname --file /tmp/requirements.yaml --prune
COPY pip_requirements.txt /tmp/pip_requirements.txt
RUN conda run -n $envname pip install -r /tmp/pip_requirements.txt
RUN echo "conda activate $envname" >> ~/.bashrc && \
    conda run -n $envname python -m ipykernel install --user --name=$envname

### Setup-for installing apex
### TODO: Make sure that the cuda driver's version matches
ENV CUDAVER cuda-10.1
ENV PATH /usr/local/$CUDAVER/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/$CUDAVER/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/$CUDAVER/lib64:$LD_LIBRARY_PATH
ENV CUDA_PATH /usr/local/$CUDAVER
ENV CUDA_ROOT /usr/local/$CUDAVER
ENV CUDA_HOME /usr/local/$CUDAVER
ENV CUDA_HOST_COMPILER /usr/bin/gcc

RUN git clone https://github.com/NVIDIA/apex.git
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /home/$UNAME

# When starting the container for the first time run:
# cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
###

ENTRYPOINT sudo service ssh start && /bin/bash
