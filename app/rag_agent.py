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
import init_qdrant

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

SYSTEM_PROMPT = """You are a medically trained assistant analyzing clinical reports.

Answer questions using only the information from the uploaded medical documents. Be thorough, precise, and cautious — clearly state if additional consultation is needed.

If a direct answer is not available in the context, infer carefully and explain your reasoning. If it's completely absent, respond with: \"I could not find that in the reports.\"

For questions related to patient health, diagnosis, risk factors, or preventive suggestions — use your knowledge only in the context of the reports provided.

If a question is unrelated to the uploaded medical reports, say: \"I cannot answer that.\"

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
