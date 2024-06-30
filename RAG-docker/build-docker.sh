#!/bin/bash

# docker build  --no-cache --network=host -t rag-llama2-custom-pdfs . 
#docker build --no-cache -t rag-chatbot .
docker build -t rag-chatbot .
