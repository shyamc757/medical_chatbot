from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

from config import QDRANT_COLLECTION_NAME, QDRANT_URL
from dotenv import load_dotenv
load_dotenv()

qdrant_client = QdrantClient(url=QDRANT_URL)
embedding_model = OpenAIEmbeddings()

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embedding_model,
    content_payload_key="text"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
    content_payload_key="text"
)

SYSTEM_PROMPT = """You are a helpful and precise medical assistant.
Answer the question using only the information from the provided context.
If the answer is not found in the context, say "I could not find that in the reports."

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
output_parser = StrOutputParser()

def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved docs into a single context block."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain: Runnable = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | output_parser
)


def get_answer(question: str) -> str:
    """Public interface to ask a question using the RAG agent."""
    return rag_chain.invoke({"question": question})
