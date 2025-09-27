import time
import logging
import asyncio
from pathlib import Path
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "medical-rag-index"

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_pinecone_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required but not set.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(
        cloud="aws",
        region=PINECONE_ENV or "us-east-1",
    )
    existing_indexes = [index["name"] for index in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=spec,
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = get_pinecone_index()

    for file in uploaded_files:
        try:
            save_path = Path(UPLOAD_DIR) / file.filename
            with open(save_path, "wb") as f:
                f.write(file.file.read())

            loader = PyPDFLoader(str(save_path))
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )

            chunks = text_splitter.split_documents(documents)

            texts = [chunk.page_content for chunk in chunks]
            ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
            metadata = [
                {
                    "text": chunk.page_content,
                    "source": file.filename,
                    "doc_id": doc_id,
                    "role": role,
                    "page": chunk.metadata.get("page", 0)
                }
                for i, chunk in enumerate(chunks)
            ]

            print(f"Embedding {len(texts)} Chunk...")

            embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)

            print("Uploading to Pinecone...")

            with tqdm(total=len(embeddings), desc="Upserting To Pinecone") as progress:
                await asyncio.to_thread(
                    index.upsert,
                    vectors=list(zip(ids, embeddings, metadata)),
                )
                progress.update(len(embeddings))
            
            print(f"Uploaded {len(embeddings)} Vectors for {file.filename} To Pinecone")
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            continue

