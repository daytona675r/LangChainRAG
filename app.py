# Load environment variables
import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


#environment variables
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
  os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

#chat model initialization
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Embeddings initialization
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector store initialization
vector_store = InMemoryVectorStore(embeddings)


