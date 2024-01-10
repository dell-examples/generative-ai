# Created by Scalers AI for Dell Inc.

# Use the official Nvidia NGC PyTorch image as the base image
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set the working directory inside the container
WORKDIR /usr/src/

# Install necessary build dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file
COPY ./requirements.txt /usr/src/requirements.txt

# Install the required Python packages from the requirements file
RUN pip install --no-cache-dir --upgrade -r /usr/src/requirements.txt

# Copy the API code to the container
COPY ./api /usr/src/api

# Set the working directory to the API directory
WORKDIR /usr/src/api

# For informative purpose only. API is available in 8000. 
EXPOSE 8000

# Specify the command to run when the container starts
CMD ["sh","run.sh"]