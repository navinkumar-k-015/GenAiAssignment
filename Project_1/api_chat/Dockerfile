FROM python:3.9-slim-buster

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /opt/program/requirements.txt

RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY model /opt/program

WORKDIR /opt/program/

RUN chmod 755 *

CMD ["uvicorn", "predictor:app", "--reload", "--host", "0.0.0.0"]

