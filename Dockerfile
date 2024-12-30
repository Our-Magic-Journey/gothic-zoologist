FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y bash imagemagick
RUN pip install poetry