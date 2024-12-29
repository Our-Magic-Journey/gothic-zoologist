FROM python:3.12-slim

RUN apt-get install bash
RUN pip install poetry