# Created by Scalers AI for Dell Inc.

# Use the official Python 3.10 image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/

# Update package lists and install curl
RUN apt-get update && apt-get install -y curl

# Copy the requirements file and the application code to the container
COPY ./requirements.txt /usr/src/requirements.txt

# Install the required Python packages from the requirements file
RUN pip install --no-cache-dir --upgrade -r /usr/src/requirements.txt

# Copy the application code to the container
COPY ./app /usr/src/app

# Set the working directory to the 'app' directory
WORKDIR /usr/src/app

# For informative purpose only. UI is available in 9000. 
EXPOSE 9000

# Specify the command to run when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]