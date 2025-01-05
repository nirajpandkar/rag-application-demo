from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
import os.path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

with open("openai-key.txt") as infile:
    openai_api_key = infile.read()
os.environ['OPENAI_API_KEY'] = openai_api_key

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()

    
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Using existing stored index")
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


query_engine = index.as_query_engine()
response = query_engine.query("""In the beginning of all the documents, you'll find numbered bullets listing out the heading of the articles. Please list all of them out here.""")
print(response)