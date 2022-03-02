FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.5.1-transformers4.12.3-gpu-py37-cu112-ubuntu18.04

COPY resources/train.py /opt/program/train.py

ENV SAGEMAKER_PROGRAM /opt/program/train.py