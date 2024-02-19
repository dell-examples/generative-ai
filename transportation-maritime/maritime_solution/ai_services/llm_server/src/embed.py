# Created by scalers.ai for Dell Inc

import argparse
import logging

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

class Embeddings:
    def __init__(self):
            """
            Initialize an Embeddings object with a tokenizer and text splitter.

            Returns:
                None
            """
            self.tokenizer = AutoTokenizer.from_pretrained(
                        "thenlper/gte-large"
            )
            self.text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                        self.tokenizer, chunk_size=700, chunk_overlap=50
                    )
            
    def build_vector_db_qa_pdf(self, docs_dir, db_path, logger):
        """
        Create vector database containing text embeddings.

        Args:
            docs_dir (str): Directory containing text documents.
            db_path (str): Path to store the vector database.

        Returns:
            None
        """
        loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)

        docs = loader.load()

        texts = self.text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
        
        self.db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
        self.db.persist()
        logger.info(f"count final {self.db._collection.count()}")

    def load_pdfs(self, folder_path, db_path, logger):
        """
        Load PDF files from a folder and build a vector database.

        Args:
            folder_path (str): Path to the folder containing PDF files.
            db_path (str): Path to store the vector database.

        Returns:
            None
        """
        self.build_vector_db_qa_pdf(folder_path, db_path, logger)


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

    # Initialize logger
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # Read arguments from command line
    args = parser.parse_args()
    context_folder = args.context
    if context_folder:
        k = Embeddings()
        k.load_pdfs(context_folder, logger)
