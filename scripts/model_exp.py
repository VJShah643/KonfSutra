from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import utils
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load MiniLM for embedding
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load CodeT5
generator_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m")


# Example config snippets
docs = [
    "bindsym $mod+Enter exec i3-sensible-terminal",
    "set -g prefix C-a",
    "map <C-w> :q<CR>",
]

docs = utils.read_file("/etc/i3/config").split("\n")

# Encode documents into vectors
doc_embeddings = retriever.encode(docs, convert_to_numpy=True)

# Create FAISS index for fast search
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Store mappings
doc_map = {i: doc for i, doc in enumerate(docs)}

def retrieve_config(query, top_k=1):
    query_embedding = retriever.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return [doc_map[idx] for idx in indices[0]]

def generate_response(query):
    retrieved_docs = retrieve_config(query)
    
    # Create input prompt with retrieved config
    prompt = f"Query: {query}\nConfig: {retrieved_docs[0]}\nAnswer:"
    
    inputs = generator_tokenizer(prompt, return_tensors="pt")
    output_ids = generator_model.generate(**inputs, max_length=50)
    response = generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response





while True:
    query = input("Enter your query: ")
    
    if query == "exit":
        break

    response = generate_response(query)

    print(response)


