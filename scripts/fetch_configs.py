import json
import os
import time
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CONFIGS_FILE = "configs.json"
OUTPUT_DIR = "data/configs"
METADATA_FILE = "data/metadata.json"
FAISS_INDEX_FILE = "data/configs_index.faiss"
CHUNK_SIZE = 500  # Characters per chunk, adjust as needed

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_configs():
    with open(CONFIGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def expand_path(path_str):
    expanded = os.path.expanduser(str(path_str))
    expanded = os.path.expandvars(expanded)
    return Path(expanded)

def expand_paths(paths):
    return [expand_path(p) for p in paths]

def get_safe_filename(original_path):
    return original_path.replace("/", "_").replace("~", "home").strip("_")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_file(file_path, output_dir, metadata, chunks_list):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        safe_name = get_safe_filename(str(file_path))
        output_path = Path(output_dir) / safe_name

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Chunk the content and store with metadata
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{safe_name}_chunk_{i}"
            metadata[chunk_id] = {
                "original_path": str(file_path),
                "backup_path": str(output_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunk_index": i,
                "chunk_text": chunk
            }
            chunks_list.append(chunk)
        print(f"✅ Saved and chunked: {output_path}")

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {str(e)}")

def crawl_directory(dir_path, output_dir, metadata, chunks_list):
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"⚠️ Directory not found or invalid: {dir_path}")
        return

    for item in dir_path.rglob("*"):
        if item.is_file():
            process_file(item, output_dir, metadata, chunks_list)

def save_metadata_and_index(metadata, chunks_list):
    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Generate embeddings and save FAISS index
    embeddings = model.encode(chunks_list)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ Saved FAISS index with {len(chunks_list)} chunks")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configs = load_configs()

    metadata = {}
    chunks_list = []
    
    for program, data in configs.items():
        for path in expand_paths(data.get("config_paths", [])):
            if not path.exists():
                print(f"⚠️ Path does not exist after expansion: {path}")
                continue
            if path.is_dir():
                print(f"📁 Crawling directory: {path}")
                crawl_directory(path, OUTPUT_DIR, metadata, chunks_list)
            elif path.is_file():
                print(f"📄 Processing file: {path}")
                process_file(path, OUTPUT_DIR, metadata, chunks_list)
            else:
                print(f"⚠️ Skipping invalid path: {path}")

    save_metadata_and_index(metadata, chunks_list)
