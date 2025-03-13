from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import argparse

SHORTCUTS_FILE = "shortcuts.json"
MODEL_NAME = "Salesforce/codegen-350M-mono"  # Lightweight model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

def load_shortcuts():
    """Load the consolidated shortcuts file."""
    with open(SHORTCUTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_keybind(query, shortcuts):
    """Query the model to extract the keybind from shortcuts."""
    shortcuts_text = "\n".join([f"{task}: {keybind}" for task, keybind in shortcuts.items()])
    prompt = f"""Given the query and the list of shortcuts below, return ONLY the keybind (e.g., 'Ctrl-b %') that matches the task in the query. If no match is found, return 'No keybind found'.

Query: {query}

Shortcuts:
{shortcuts_text}

Keybind:"""

    inputs = TOKENIZER(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    keybind_start = response.find("Keybind:")
    return response[keybind_start + len("Keybind:"):].strip() if keybind_start != -1 else response.strip()

def main():
    print("Loading shortcuts...")
    shortcuts = load_shortcuts()
    print("Ready! Type your question (or 'quit' to exit):")

    while True:
        query = input("> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        keybind = get_keybind(query, shortcuts)
        print(f"\nKeybind:\n{keybind}\n")

if __name__ == "__main__":
    main()
