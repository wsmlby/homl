# This Dockerfile builds the final HoML Server image for CPU.
# It layers the HoML server code on top of a pre-built vLLM CPU base image.
FROM ghcr.io/wsmlby/homl-vllm-cpu-base:latest


# Set the working directory to homl_server
WORKDIR /app/homl_server


# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Optionally install from pyproject.toml if needed
# COPY pyproject.toml .
# RUN pip install .


# Copy our application source code
COPY ./homl_server ./

ARG HOML_SERVER_VERSION=dev
ENV HOML_SERVER_VERSION=$HOML_SERVER_VERSION

# Start the server directly from main.py
CMD ["python3", "-u", "main.py"]
