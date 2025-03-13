from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os

# Model configurations
MODEL_CONFIGS = {
    #"deepseek-1.3b": {
    #    "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
    #    "quantization": None,
    #    "description": "DeepSeek Coder 1.3B - Lightweight, code-optimized",
    #    "vram_est": "3GB"
    #},
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
    """Authenticate with Hugging Face using an API token."""
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

def download_model(model_key, config):
    """Download a model and its tokenizer."""
    print(f"\nDownloading {config['description']}...")
    
    # Setup quantization if required
    quantization_config = None
    if config["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Using 4-bit quantization.")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        print(f"Tokenizer for {model_key} downloaded.")
        
        # Download model weights
        model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None
        )
        if not quantization_config:
            # If no quantization, move to CPU to avoid GPU memory usage during download
            model = model.to("cpu")
        print(f"Model weights for {model_key} downloaded.")
        
    except Exception as e:
        print(f"Error downloading {model_key}: {str(e)}")

def main(token=None):
    """Download all models in MODEL_CONFIGS."""
    # Authenticate once
    authenticate_huggingface(token)
    
    # Download each model
    for model_key, config in MODEL_CONFIGS.items():
        download_model(model_key, config)
    
    print("\nAll models downloaded successfully! They are stored in ~/.cache/huggingface/hub.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download models from MODEL_CONFIGS")
    parser.add_argument("--token", type=str, help="Hugging Face API token", default=None)
    args = parser.parse_args()

    main(args.token)
