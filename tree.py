import os
from pathlib import Path

# Expanded ignore list to catch all virtual env variants
IGNORE_DIRS = {'.venv', '.venv311', 'venv', 'env', 'Lib', 'Scripts', 'Include', '.git', '__pycache__', '.idea', '.vscode'}
IGNORE_EXTS = {'.pyc', '.parquet', '.onnx', '.bin', '.json', '.csv', '.whl', '.pyd', '.dll', '.exe', '.chm', '.txt'}

def generate_tree(dir_path, file_obj, prefix=""):
    path = Path(dir_path)
    
    entries = sorted(
        [e for e in path.iterdir() if e.name not in IGNORE_DIRS and e.suffix.lower() not in IGNORE_EXTS],
        key=lambda e: (e.is_file(), e.name.lower())
    )
    
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        file_obj.write(f"{prefix}{connector}{entry.name}\n")
        
        if entry.is_dir():
            new_prefix = prefix + ("    " if is_last else "│   ")
            generate_tree(entry, file_obj, new_prefix)

# Write directly to a file to bypass terminal truncation
output_file = "project_tree_clean.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"{Path.cwd().name}/\n")
    generate_tree(".", f)
    
print(f"Success! Clean tree saved to {output_file}")