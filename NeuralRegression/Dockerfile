# base image
# a little overkill but need it to install dot cli for dtreeviz
# build off the base ubuntu image
FROM ubuntu:18.04

# ubuntu installing - python, pip, graphviz
RUN apt-get update &&\
    apt-get install python3.8 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y

RUN pip3 install Cython &&\
    pip3 install pyarrow==0.9.0&&\ 
    pip3 install -U --force-reinstall pip
# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /NeuralRegression

# copy over requirements
#It is typically not recommended to copy all files to the image the way we've done above 
#(particularly if you have large files). However, since this is a small example, it won't cause any issues for us.

COPY requirements.txt ./requirements.txt

# install pip then packages
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# cmd to launch app when container is run
CMD streamlit run App/main.py

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