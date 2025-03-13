import os
import subprocess
import json
import time
from pathlib import Path

MAN_DIR = "data/man_pages"
METADATA_FILE = "data/metadata.json"
PROGRAMS_FILE = "programs.json"

# Ensure metadata is loaded
def load_metadata():
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save metadata to a JSON file
def save_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

# Load the programs list from programs.json
def load_programs():
    try:
        with open(PROGRAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Helper function to generate unique filenames for man pages
def get_safe_filename(program_name, section):
    """Creates a unique filename for each man page based on the program name and section."""
    return f"{program_name}_{section}".replace("/", "_").strip("_")

# Extract and save the man page content, and track metadata
def extract_and_save_man_page(file, section, metadata, program_name):
    """Fetches a man page from a file, saves it and updates metadata."""
    output_path = Path(MAN_DIR) / f"{program_name}_{section}.txt"

    # Only process if the file doesn't already exist
    if not output_path.exists():
        try:
            # Use zcat to extract gzipped man page content
            result = subprocess.run(["zcat", str(file)], capture_output=True, text=True, check=True)

            # Save the extracted content to a text file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            # Update metadata
            metadata[f"{program_name}_{section}"] = {
                "program_name": program_name,
                "section": section,
                "backup_path": str(output_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            print(f"✅ Extracted: {output_path}")

        except subprocess.CalledProcessError:
            print(f"⚠️ Error extracting {file}")
        except Exception as e:
            print(f"⚠️ Unexpected error: {str(e)}")

# Function to extract all man pages from the man directories
def extract_man_pages():
    os.makedirs(MAN_DIR, exist_ok=True)

    # Load the programs from the JSON file
    programs = load_programs()
    
    # Man directories (can be extended with other sections)
    man_dirs = [
        "/usr/share/man/man1/",  # Section 1: User commands
        "/usr/share/man/man2/",  # Section 2: System calls
        "/usr/share/man/man3/",  # Section 3: Library calls
        "/usr/share/man/man4/",  # Section 4: Special files
        "/usr/share/man/man5/",  # Section 5: File formats
        "/usr/share/man/man6/",  # Section 6: Games
        "/usr/share/man/man7/",  # Section 7: Miscellaneous
        "/usr/share/man/man8/"   # Section 8: System administration
    ]

    # Load previous metadata to keep track of existing files
    metadata = load_metadata()

    # Traverse all man directories and extract .gz man files
    for dir_path in man_dirs:
        for file in Path(dir_path).glob("*.gz"):
            cmd_name = file.stem.split('.')[0]

            # Check if the cmd_name exists in either the main programs or the misc man pages list
            if cmd_name in programs or cmd_name in programs.get("misc", {}).get("misc_man_pages", []):
                # Extract man pages from relevant sections (e.g., tmux(1), tmux(5), etc.)
                for section in ["1"]:#, "5", "8"]:
                    extract_and_save_man_page(file, section, metadata, cmd_name)

    # Save the updated metadata
    save_metadata(metadata)

if __name__ == "__main__":
    extract_man_pages()

