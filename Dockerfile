# Filename: Dockerfile

#FROM node:10-alpine
#WORKDIR /usr/src/app
#COPY package*.json ./
#RUN npm install
#COPY . .

FROM ubuntu:latest

#ENV DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev git wget unzip python3-opencv \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Setup a spot for the api code
RUN mkdir -p /usr/local/src/heddex
WORKDIR  /usr/local/src/heddex

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
## Install poppler
#RUN apt-get update && apt-get -y install poppler-utils

# Copy over the source code
COPY run_download_model.sh run_download_model.sh
RUN bash run_download_model.sh

COPY . /usr/local/src/heddex
RUN python setup.py install

WORKDIR  /usr/local/src/heddex/tools
EXPOSE 8000
ENTRYPOINT [ "uvicorn" ]
CMD ["inference_server:app", "--host", "0.0.0.0", "--reload", "--port", "8080"]
