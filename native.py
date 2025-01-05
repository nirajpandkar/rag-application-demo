import os
import os.path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Read the api key
# with open("openai-key.txt") as infile:
#     openai_api_key = infile.read()
# os.environ['OPENAI_API_KEY'] = openai_api_key

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set the LLM to be used for querying
Settings.llm = Ollama(model="llama3.2", request_timeout=3600.0,)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    print("Createing store")
    # load the documents and create the index
    documents = SimpleDirectoryReader("data", recursive=True, required_exts=[".md"]).load_data()

    
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Using existing stored index")
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


query_engine = index.as_query_engine()
response = query_engine.query("""Write a function in PURE language to return a difference between two StrictDate dates""")
print(response)