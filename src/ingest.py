import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

BATCH_SIZE = 10
RATE_LIMIT_WAIT_SECONDS = 60


def ingest_pdf():
    documents = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    store = PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL")),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        try:
            store.add_documents(batch)
        except Exception:
            print(f"Limite de requisições atingido. Aguardando {RATE_LIMIT_WAIT_SECONDS}s...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            store.add_documents(batch)

    print(f"Ingestão concluída: {len(chunks)} chunks salvos.")


if __name__ == "__main__":
    ingest_pdf()
