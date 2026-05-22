import os
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from .prompt_templates import rag_prompt
from langchain_community.tools.tavily_search import TavilySearchResults
import config.env  # Initialize environment variables

# Get env variables
index_name = os.environ.get("index_name", "healthguru").strip()
groq_model = os.environ.get("groq_model", "llama3-70b-8192").strip()

print(f"Using Groq model: {groq_model}")
print(f"Using index: {index_name}")

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
tavily_search = TavilySearchResults(max_results=3)

llm = ChatGroq(model=groq_model, temperature=0.3, max_tokens=1024)

rag_chain = rag_prompt | llm

__all__ = ["llm", "retriever", "rag_chain", "tavily_search"]
