#!/usr/bin/env python3
"""
LisanClean Root Extractor

Parsed lisanclean.json to create a mapping of Arabic Words -> Roots.
This map will be used by Quran WhisperX to enforce morphological boundaries during alignment.
"""
import json
import re
from pathlib import Path

LISAN_PATH = '/home/absolut7/Documents/ihya_data_export/core_data/raw/lisanclean.json'
OUTPUT_PATH = Path('public/data/lisan_roots.json')

def extract_root(explanation):
    """
    Extracts root from explanation string.
    Format is typically: "@ROOT: description..." or just starts with word.
    Actually looking at the raw data:
    "@أَبأ: ..." -> Root is "أبأ"
    "@أَتأ: ..." -> Root is "أتأ"
    """
    if explanation.startswith('@'):
        # Extract content between @ and :
        match = re.match(r'@([^:]+):', explanation)
        if match:
            return match.group(1).strip()
    return None

def main():
    print(f"Loading LisanClean from {LISAN_PATH}...")
    try:
        with open(LISAN_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load Lisan: {e}")
        return

    print(f"Entries found: {len(data)}")
    
    root_map = {}
    
    for entry in data:
        word = entry.get('word', '').strip()
        explanation = entry.get('explanation', '')
        
        # In this dataset, 'word' seems to be the ROOT itself for the entry header
        # But let's check if there are derived forms.
        # Actually examining the data:
        # { "word": "أبأ", "explanation": "@أَبأ: ..." }
        # This dataset seems to be a ROOT dictionary, not a word-form dictionary.
        # i.e. it lists roots and defines them.
        
        # Ideally we need: Word-Form -> Root (e.g. "kitab" -> "ktb")
        # If this dataset ONLY has roots, we might need a morphological analyzer + this dataset.
        
        # Let's just index what we have: Root -> Definition
        root = extract_root(explanation)
        if not root:
            root = word # Fallback
            
        if root:
             root_map[root] = explanation[:100] + "..." # Store brief def
            
    print(f"Extracted {len(root_map)} unique roots.")
    
    # Save extracted roots
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(root_map, f, ensure_ascii=False, indent=2)
        
    print(f"Saved root map to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
