FROM openjdk:slim

MAINTAINER  George Makridis <gmakridis@unipi.gr>

COPY --from=python:3.7 / /

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip install setuptools
# RUN pip install kafka-python

COPY . /processor
WORKDIR /processor

RUN pip install -r requirements.txt

#EXPOSE 5000

# Add docker-compose-wait tool -------------------
ENV WAIT_VERSION 2.7.2
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/$WAIT_VERSION/wait /wait
RUN chmod +x /wait

#CMD sleep infinity

ENTRYPOINT ["python", "/processor/main.py", "$@", "--"]
