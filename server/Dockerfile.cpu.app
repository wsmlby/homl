# This Dockerfile builds the final HoML Server image for CPU.
# It layers the HoML server code on top of a pre-built vLLM CPU base image.
ARG BASE_IMAGE=homl/vllm-cpu:latest
FROM ${BASE_IMAGE}


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


# Start the server directly from main.py
CMD ["python3", "-u", "main.py"]
