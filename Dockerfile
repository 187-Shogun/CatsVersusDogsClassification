FROM python:3.8.8-slim-buster

# Copy jobs into working directory:
COPY . /app
WORKDIR /app

# Install luigi:
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set env variables:
ENV PYTHONPATH /app
ENV LUIGI_CONFIG_PATH /app/luigi.cfg