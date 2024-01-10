# use a python container as a starting point
FROM python:3.10-slim

# install dependencies of interest
RUN python -m pip install rasa==3.6.8

WORKDIR /
# Copy the application code to the container
COPY ./app /app

# set workdir and copy data files from disk
# note the latter command uses .dockerignore
WORKDIR /app
ENV HOME=/app

# train a new rasa model
RUN rasa train nlu

# train model
RUN rasa train

# set the user to run, don't run as root
USER 1001

EXPOSE 5055
EXPOSE 5005

CMD ["sh","run.sh"]