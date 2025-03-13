import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import textwrap

CONFIG_DIR = "data/configs"
MAN_DIR = "data/man_pages"
FAISS_INDEX_PATH = "faiss_index/index"

retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def load_text_files(directory, chunk_size=300):
    texts, paths, metadata = [], [], []
    for file in Path(directory).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            chunks = chunk_text(content, chunk_size)
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                paths.append(str(file))
                metadata.append(f"{str(file)}_chunk_{i}")
    return texts, paths, metadata

def build_faiss_index():
    texts, paths, metadata = [], [], []

    # Load configs & man pages
    config_texts, config_paths, config_metadata = load_text_files(CONFIG_DIR)
    man_texts, man_paths, man_metadata = load_text_files(MAN_DIR)

    texts.extend(config_texts)
    texts.extend(man_texts)
    paths.extend(config_paths)
    paths.extend(man_paths)
    metadata.extend(config_metadata)
    metadata.extend(man_metadata)

    if not texts:
        print("⚠️ No documents to index!")
        return

    # Compute embeddings
    embeddings = retriever.encode(texts, convert_to_numpy=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(f"{FAISS_INDEX_PATH}_map.txt", "w", encoding="utf-8") as f:
        for i, (path, meta) in enumerate(zip(paths, metadata)):
            f.write(f"{i}\t{path}\t{meta}\n")

    print(f"✅ FAISS index built and saved!")

if __name__ == "__main__":
    build_faiss_index()
