import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
import argparse
import os

# File paths
METADATA_FILE = "data/metadata.json"
FAISS_INDEX_FILE = "data/configs_index.faiss"

# Load embedding model for retrieval
RETRIEVAL_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Model configurations
MODEL_CONFIGS = {
    "codellama-7b": {
        "name": "codellama/CodeLLaMA-7b-hf",
        "quantization": "4bit",
        "description": "CodeLLaMA 7B (4-bit quantized) - Good for code and configs",
        "vram_est": "3.5GB"
    },
    "starcoder": {
        "name": "bigcode/starcoder",
        "quantization": None,
        "description": "StarCoder 1B - Lightweight, fast, code-focused",
        "vram_est": "2GB"
    },
    "codegen-2b": {
        "name": "Salesforce/codegen-2B-mono",
        "quantization": None,
        "description": "CodeGen 2B Mono - Balanced size and capability",
        "vram_est": "4GB"
    },
    "phi-1.5": {
        "name": "microsoft/phi-1_5",
        "quantization": None,
        "description": "Phi-1.5 1.3B - Small, efficient, technical",
        "vram_est": "2.6GB"
    },
    "codellama-13b": {
        "name": "codellama/CodeLLaMA-13b-hf",
        "quantization": "4bit",
        "description": "CodeLLaMA 13B (4-bit quantized) - Powerful but VRAM-heavy",
        "vram_est": "6.5GB"
    },
    "deepseek-1.3b": {
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "quantization": None,
        "description": "DeepSeek Coder 1.3B - Lightweight, code-optimized",
        "vram_est": "3GB"
    },
    "codet5p-770m": {
        "name": "Salesforce/codet5p-770m",
        "quantization": None,
        "description": "CodeT5+ 770M - Efficient for code understanding",
        "vram_est": "2GB"
    },
    "starcoderbase-1b": {
        "name": "bigcode/starcoderbase-1b",
        "quantization": None,
        "description": "StarCoderBase 1B - Lightweight, no gating",
        "vram_est": "2.5GB"
    },
    "polycoder-160m": {
        "name": "NinedayWang/PolyCoder-160M",
        "quantization": None,
        "description": "PolyCoder 160M - Ultra-light, fast",
        "vram_est": "1GB"
    },
    "codegen-350m": {
        "name": "Salesforce/codegen-350M-mono",
        "quantization": None,
        "description": "CodeGen 350M Mono - Small and efficient",
        "vram_est": "1.5GB"
    }
}

def authenticate_huggingface(token=None):
    """Authenticate with Hugging Face using an API token from env var, argument, or input."""
    # Check environment variable first
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        print("Using HF_TOKEN from environment variable.")
        login(env_token)
    elif token:
        print("Using provided token.")
        login(token)
    else:
        token = input("Enter your Hugging Face API token: ").strip()
        login(token)
    print("Successfully authenticated with Hugging Face.")

def load_model(model_key):
    """Load the specified model based on its configuration."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_key}' not found in MODEL_CONFIGS")

    config = MODEL_CONFIGS[model_key]
    print(f"Loading {config['description']}...")

    # Setup quantization if required
    quantization_config = None
    if config["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif config["quantization"] == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["name"],
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None
    )
    if not quantization_config:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model

def load_metadata_and_index():
    """Load metadata and FAISS index from files."""
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        index = faiss.read_index(FAISS_INDEX_FILE)
        return metadata, index
    except FileNotFoundError as e:
        print(f"Error: Could not find required files - {e}")
        exit(1)

def query_configs(query, metadata, index, top_k=3):
    """Search the FAISS index for relevant config chunks."""
    query_embedding = RETRIEVAL_MODEL.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        chunk_id = list(metadata.keys())[idx]
        chunk_data = metadata[chunk_id]
        results.append({
            "text": chunk_data["chunk_text"],
            "path": chunk_data["original_path"],
            "score": float(distances[0][i])
        })
    return results

def generate_answer(query, results, tokenizer, model):
    """Generate a natural language answer using the selected model."""
    if not results:
        return "Sorry, I couldn't find relevant information in your configs."

    context = "\n".join([f"File: {r['path']}\nContent: {r['text']}" for r in results])
    prompt = f"""You are an expert in Linux configuration files. Based on the query and config file excerpts below, extract and return ONLY the keybind (e.g., 'Ctrl-b %') for the specified task. Do not include any explanation or additional text. If no keybind is found, return 'No keybind found'.
____________________________________________________
Query: {query}
____________________________________________________
Config Excerpts:
{context}
____________________________________________________
Keybind:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #answer_start = response.find("Answer:")
    #return response[answer_start + len("Answer:"):].strip() if answer_start != -1 else response.strip()
    keybind_start = response.find("Keybind:")
    answer = response[keybind_start + len("Keybind:"):].strip()
    # Strip any extra characters or text beyond the first line
    answer = answer.split('\n')[0].strip()
    return answer

def list_models():
    """Print available models and their details."""
    print("\nAvailable models:")
    for key, config in MODEL_CONFIGS.items():
        print(f"- {key}: {config['description']} (Est. VRAM: {config['vram_est']})")

def main(model_key=None, token=None):
    """Main function with model selection and authentication."""
    # Authenticate with Hugging Face
    authenticate_huggingface(token)

    # If no model specified, prompt user
    if not model_key:
        list_models()
        model_key = input("\nEnter the model key to use (e.g., 'codellama-7b'): ").strip()
        if model_key not in MODEL_CONFIGS:
            print(f"Invalid model key. Available options: {', '.join(MODEL_CONFIGS.keys())}")
            return

    # Load the selected model
    tokenizer, model = load_model(model_key)
    print("Loading configuration database...")
    metadata, index = load_metadata_and_index()
    print(f"Ready with {MODEL_CONFIGS[model_key]['description']}! Type your question (or 'quit' to exit):")

    while True:
        query = input("> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        results = query_configs(query, metadata, index)
        print("\nTop results:", results)
        answer = generate_answer(query, results, tokenizer, model)
        print(f"\nAnswer after generation:\n{answer}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query Linux configs with different LLMs")
    parser.add_argument("--model", type=str, help="Model key (e.g., 'codellama-7b')", default=None)
    parser.add_argument("--token", type=str, help="Hugging Face API token", default=None)
    args = parser.parse_args()

    main(args.model, args.token)
