FROM python:3.12.12

RUN mkdir /app
WORKDIR /app

COPY ./* /app/
