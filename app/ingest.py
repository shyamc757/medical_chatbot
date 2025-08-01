from typing import List
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm

from extract import process_files
from config import QDRANT_COLLECTION_NAME, QDRANT_URL

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

qdrant = QdrantClient(url=QDRANT_URL)
embedder = OpenAIEmbeddings()

def chunk_text(text: str) -> List[str]:
    """Split raw text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def ingest_reports(input_dir: str):
    """
    Process all medical reports in a folder, extract content, and ingest into Qdrant.

    Args:
        input_dir (str): Path to folder containing medical reports.
    """
    if qdrant.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant.delete_collection(QDRANT_COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    reports = process_files(input_dir)
    for report in tqdm(reports, desc="Ingesting reports"):
        filename = report["filename"]
        timestamp = report["timestamp"]
        raw_text = report["raw_text"]
        parsed_fields = report["parsed_fields"]

        chunks = chunk_text(raw_text)
        embeddings = embedder.embed_documents(chunks)

        points = []

        if parsed_fields:
            summary_text = "\n".join([
                f"{k.replace('_', ' ').title()}: {v}" for k, v in parsed_fields.items()
            ])
            summary_vector = embedder.embed_query(summary_text)
            summary_payload = {
                "text": summary_text,
                "metadata": {
                    "filename": filename,
                    "chunk_id": f"{filename}_summary",
                    "timestamp": timestamp
                }
            }

            points.append(PointStruct(id=str(uuid4()), vector=summary_vector, payload=summary_payload))

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{filename}_{i}"
            payload = {
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "timestamp": timestamp
                }
            }

            points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))

        qdrant.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        print(f"Ingested {len(points)} chunks from {filename}")