# Created by scalers.ai for Dell Inc
import os

import torch
from fastapi import FastAPI, Response, status
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

app = FastAPI()


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def get_docsearch():
    """Return text embeddings of context."""
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma(
        persist_directory="./db/", embedding_function=embeddings
    )
    return docsearch


def create_hf_pipeline(model_name):
    """Create HF pipeline for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=HF_TOKEN
    )

    # Define a mapping from precision to configuration
    precision_config = {
        "BF16": (torch.bfloat16, False, False, DEVICE),
        "FP16": (torch.float16, False, False, DEVICE),
        "FP32": (torch.float32, False, False, DEVICE),
        "FP4": (None, True, False, None),
        "INT8": (None, False, True, None),
    }

    # Get the configuration based on the selected precision
    torch_dtype, load_in_4bit, load_in_8bit, device = precision_config.get(
        PRECISION
    )

    # Load the model with the selected configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_auth_token=HF_TOKEN,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        device=device,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # # Create a custom llm using HF pipeline wrapper
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    return hf_pipeline


def create_prompt():
    """Prompt template for LLM."""

    prompt_template = """
    <s>[INST] <<SYS>> {{ You are a AI chatbot having a conversation with a human.
    Given the following has two parts.
    First part is a extracted parts of a long document.
    Second part is the human's question..
    Your goal is to complete the conversation by answering the human's question.

    {context}

    Question: {question}
    Answer: }}
    <</SYS>>
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return prompt


def create_qa():
    """Create question answering chain."""
    docsearch = get_docsearch()
    chain_type_kwargs = {"prompt": PROMPT}
    retriever = docsearch.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(
        llm=HF_PIPELINE,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa


@app.get("/qa/predict", status_code=200)
def process_query(query: str, http_response: Response):
    """Predic api call for text generation."""
    if QA:
        answer = QA.run(query)
        print(answer)
        return str(answer)
    else:
        http_response.status_code = status.HTTP_400_BAD_REQUEST
        response = {"response": "Context not set"}
        return response


@app.get("/health")
async def health_check():
    """Health checkup API."""
    return {"status": "healthy"}


MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
DEVICE = os.environ.get("DEVICE", "cuda")
PRECISION = os.environ.get("PRECISION", "BF16")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 250))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
PROMPT = create_prompt()
HF_PIPELINE = create_hf_pipeline(MODEL_NAME)
QA = create_qa()
