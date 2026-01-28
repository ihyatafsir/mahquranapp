#!/usr/bin/env python3
"""
Simple Direct Copy - Use Original Timing Data As-Is

The user wants to use the original timing data exactly as-is without any 
calculations or modifications. This script:
1. Copies original timing data directly
2. Only adds wordIdx based on timing gaps (for word highlighting)
3. Preserves ALL original timing values (start, end, duration)
"""

import json
import shutil
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        if content.startswith('\ufeff'):
            content = content[1:]
        return json.loads(content)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_word_indices(timing_data):
    """
    Add wordIdx based on timing gaps (>50ms = new word).
    Does NOT modify any timing values.
    """
    word_idx = 0
    for i, entry in enumerate(timing_data):
        if i > 0:
            prev_end = timing_data[i-1].get('end', 0)
            curr_start = entry.get('start', 0)
            gap = curr_start - prev_end
            if gap > 50:  # 50ms gap = new word
                word_idx += 1
        entry['wordIdx'] = word_idx
    return timing_data

def process_surah(surah_num, base_path):
    """Process a single surah - direct copy with wordIdx."""
    original_path = base_path / f'abdul_basit_original/letter_timing_{surah_num}.json'
    output_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    
    if not original_path.exists():
        return None, f"Original not found: {original_path}"
    
    # Load original data
    data = load_json(original_path)
    
    # Filter BOM entries but keep everything else
    data = [e for e in data if e.get('char', '') != '\ufeff']
    
    # Add wordIdx based on timing gaps
    data = add_word_indices(data)
    
    # Save directly - NO modifications to timing values
    save_json(output_path, data)
    
    return {
        'surah': surah_num,
        'entries': len(data),
        'first_start': data[0]['start'] if data else 0,
        'last_end': data[-1]['end'] if data else 0
    }, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Direct copy timing data')
    parser.add_argument('--surah', type=int, help='Single surah')
    args = parser.parse_args()
    
    base_path = Path('public/data')
    
    if args.surah:
        surahs = [args.surah]
    else:
        surahs = range(1, 115)
    
    for surah_num in surahs:
        result, error = process_surah(surah_num, base_path)
        if error:
            print(f"Surah {surah_num}: ERROR - {error}")
        else:
            print(f"Surah {surah_num}: {result['entries']} entries, "
                  f"{result['first_start']/1000:.2f}s - {result['last_end']/1000:.2f}s")
    
    print("\nDone! Original timing preserved exactly.")

if __name__ == '__main__':
    main()
