FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.5.1-transformers4.12.3-cpu-py37-ubuntu18.04

COPY resources/*.* /

RUN pip install flask
RUN pip install gunicorn
RUN pip install --upgrade gevent
RUN pip install pandas

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV WORKDIR /

ENTRYPOINT ["python", "/serve.py"]
