#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def fix_abdul_basit_timing(surah_num):
    path = Path(f'public/data/abdul_basit/letter_timing_{surah_num}.json')
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out BOM or empty chars
    original_len = len(data)
    cleaned = [entry for entry in data if entry['char'] != '\ufeff']
    
    if len(cleaned) < original_len:
        print(f"Removed {original_len - len(cleaned)} BOM/bad entries.")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print("File saved.")
    else:
        print("No BOM found.")

if __name__ == "__main__":
    fix_abdul_basit_timing(1)
