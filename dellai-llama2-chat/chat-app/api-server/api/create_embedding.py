# Created by scalers.ai for Dell Inc
import argparse
import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer


def build_vector_db_qa_pdf(file_names):
    """Create vector database containing text embeddings."""
    for pdf_file_name in file_names:
        model_name = os.environ.get(
            "MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf"
        )
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=hf_token
        )
        loader = UnstructuredPDFLoader(pdf_file_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=1000, chunk_overlap=0
        )
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        # print(texts)
        for i in range(len(texts)):
            texts[i].metadata = {
                "pdf-file-name": f"{pdf_file_name}",
                "part": str(i),
            }

        database = Chroma.from_documents(
            texts, embeddings, persist_directory="./db/"
        )
        database.persist()
        print(f"Processed file [{pdf_file_name}]")


def load_pdfs(folder_path):
    """Load PDF files from docs folder."""
    file_names = []
    for filename in os.listdir(folder_path):
        filename = os.path.join(folder_path, filename)
        file_names.append(filename)
    print(file_names)
    build_vector_db_qa_pdf(file_names)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding argument
    parser.add_argument(
        "-c",
        "--context",
        default="docs",
        help="Pass folder containing PDF files to set context",
    )

    # Read arguments from command line
    args = parser.parse_args()
    context_folder = args.context
    if context_folder:
        load_pdfs(context_folder)
