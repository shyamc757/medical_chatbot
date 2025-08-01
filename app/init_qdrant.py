from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import QDRANT_COLLECTION_NAME, QDRANT_URL
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=QDRANT_URL)

if not client.collection_exists(QDRANT_COLLECTION_NAME):
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,  # OpenAI embedding vector size
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{QDRANT_COLLECTION_NAME}' created.")
else:
    print(f"ℹ️ Collection '{QDRANT_COLLECTION_NAME}' already exists.")
