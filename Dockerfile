FROM ubuntu:latest

# ubuntu installing - python, pip
RUN apt-get update &&\
    apt-get install python3.10 -y &&\
    apt-get install python3-pip -y &&\
    python3 --version &&\ 
    pip install astropy

# exposing default port for streamlit
EXPOSE 8501

# Install base utilities
RUN apt-get update && \
    apt-get install -y wget &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
 /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -c conda-forge pygmo pygmo_plugins_nonfree pykep

# making directory of app
WORKDIR /streamlit-docker

# copy over requirements
COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip uninstall keras &&\
    pip install keras --upgrade
RUN pip install -r requirements.txt

# copying all files over
ADD ./App ./app/

# cmd to launch app when container is run
CMD streamlit run app/main.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'   