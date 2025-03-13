import faiss
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

FAISS_INDEX_PATH = "faiss_index/index"
FAISS_INDEX_MAP_PATH = "faiss_index/index_map.txt"
CONFIG_DIR = "data/configs"
MAN_DIR = "data/man_pages"

retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load metadata (paths to documents and chunk metadata)
def load_metadata():
    metadata_map = {}
    with open(FAISS_INDEX_MAP_PATH, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx, path, meta = line.strip().split('\t')
            metadata_map[int(idx)] = {"path": path, "meta": meta}
    return metadata_map

metadata_map = load_metadata()

# Load text files (configs and man pages) - We are not reloading the texts here since they're already indexed
def load_text_files(directory):
    texts = []
    for file in Path(directory).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            texts.extend(chunk_text(content))  # Assuming chunk_text is used while indexing
    return texts

# Function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

# Load all texts from configs and man pages
config_texts = load_text_files(CONFIG_DIR)
man_texts = load_text_files(MAN_DIR)
texts = config_texts + man_texts  # Combine both sets of texts

def retrieve_config(query, top_k=15):
    query_embedding = retriever.encode([query], convert_to_numpy=True)

    # Perform the search on the FAISS index
    _, indices = index.search(query_embedding, top_k)

    # Ensure there are valid indices returned
    if len(indices[0]) == 0:
        print("⚠️ No matching results found.")
        return [], []

    # Retrieve the corresponding chunked documents and metadata
    retrieved_docs = [texts[idx] for idx in indices[0]]
    retrieved_metadata = [metadata_map[idx] for idx in indices[0]]

    return retrieved_docs, retrieved_metadata

# Sample query to demonstrate the retrieval
def main():
    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        retrieved_docs, retrieved_metadata = retrieve_config(query)

        # Display the results
        for i, (doc, metadata) in enumerate(zip(retrieved_docs, retrieved_metadata)):
            print(f"\nResult {i+1}:")
            print(f"Source: {metadata['path']}")
            print(f"Metadata: {metadata['meta']}")
            print(f"Document: {doc}\n")

if __name__ == "__main__":
    main()

