# Created by scalers.ai for Dell Inc
import os
from embed import Embeddings
import logging

import torch
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
import markdown
from fastapi.responses import JSONResponse, HTMLResponse
import time
import tempfile

app = FastAPI()

origins = [ "*" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize logger
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

def get_docsearch(temp_dir):
    """
    Return a document search retriever.

    Args:
        temp_dir (str): The temporary directory path.

    Returns:
        Retriever: A retriever object for document search.
    """
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large", model_kwargs=model_kwargs)
    docsearch = Chroma(
        persist_directory=temp_dir, embedding_function=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 1})
    return retriever


def create_hf_pipeline(model_name):
    """
    Create a Hugging Face pipeline for text generation.

    Args:
        model_name (str): The name of the pre-trained model to use.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        num_beams=5,
        do_sample=True,
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.01, torch_dtype=torch.bfloat16, max_new_tokens=256)
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf


def create_prompt():
    """
    Create a prompt template for Language Model (LLM).

    Returns:
        PromptTemplate: A template for the prompt.
    """

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    {question}"""

    map_prompt = PromptTemplate.from_template(template=prompt_template)
    return map_prompt

def create_qa(temp_dir):
    """
    Create a question answering chain.

    Args:
        temp_dir (str): The directory to store temporary files.

    Returns:
        Chain: The question answering chain.
    """
    retriever = get_docsearch(temp_dir)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs[:1])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | HF_PIPELINE
        | StrOutputParser()
    )

    return rag_chain

@app.get("/qa/predict", status_code=200)
async def process_query(query: str, http_response: Response):
    """
    Predict API call for text generation.

    Args:
        query (str): The query for text generation.
        http_response (Response): The HTTP response object.

    Returns:
        Union[HTMLResponse, dict]: The response containing the generated text or an error message.
    """
    global logger
    logger.info(query)
    temp_dir = tempfile.TemporaryDirectory()
    embed_obj.load_pdfs("/src/voyage_text/events/", temp_dir.name, logger)
    QA = create_qa(temp_dir.name)

    if QA:
        start = time.time()
        answer = QA.invoke(query)
        end = time.time()
        logger.info(end-start)
        logger.info(answer)
        ret = markdown.markdown(answer)
        temp_dir.cleanup()
        return HTMLResponse(content=ret)
    else:
        http_response.status_code = status.HTTP_400_BAD_REQUEST
        response = {"response": "Context not set"}
        temp_dir.cleanup()
        return response

@app.get("/qa/conclusion", status_code=200)
async def process_conclusion(query: str, http_response: Response):
    """Predict API call for generating the conclusion based on the provided query.

    Args:
        query (str): The query for generating the conclusion.
        http_response (Response): The HTTP response object.

    Returns:
        Union[HTMLResponse, dict]: The response containing the generated conclusion text or an error message.
    """
    global logger  
    logger.info(query)
    temp_dir = tempfile.TemporaryDirectory()
    embed_obj.load_pdfs('/src/voyage_text/events_conclusion/', temp_dir.name, logger)
    QA = create_qa(temp_dir.name)

    if QA:
        start = time.time()
        answer = QA.invoke(query)
        end = time.time()
        logger.info(end-start)
        logger.info(answer)
        ret = markdown.markdown(answer)
        temp_dir.cleanup()
        return HTMLResponse(content=ret)
    else:
        http_response.status_code = status.HTTP_400_BAD_REQUEST
        response = {"response": "Context not set"}
        temp_dir.cleanup()
        return response


# Global Variables
MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
DEVICE = os.environ.get("DEVICE", "cuda")
PRECISION = os.environ.get("PRECISION", "BF16")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 256))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0))
PROMPT = create_prompt()
HF_PIPELINE = create_hf_pipeline(MODEL_NAME)
embed_obj = Embeddings()
