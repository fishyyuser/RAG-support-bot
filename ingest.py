import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["USER_AGENT"] = "RAG_SUPPORT_BOT"
from logger import setup_logger

# Initialize logger
logger = setup_logger()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/documentation.pdf"
VECTORSTORE_PATH = "vectorstore"


def ingest():
    try:
        logger.info("Starting ingestion pipeline...")

        # Step 1: Load PDF
        if not os.path.exists(PDF_PATH):
            logger.error(f"PDF not found at path: {PDF_PATH}")
            return

        logger.info("Loading PDF...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")

        # Step 2: Split into chunks
        logger.info("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Create embeddings
        logger.info("Initializing embeddings model...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Step 4: Store in FAISS
        logger.info("Creating FAISS vector store...")
        db = FAISS.from_documents(chunks, embeddings)

        logger.info("Saving vector store locally...")
        db.save_local(VECTORSTORE_PATH)

        logger.info("Ingestion completed successfully.")

    except Exception as e:
        logger.exception(f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    ingest()