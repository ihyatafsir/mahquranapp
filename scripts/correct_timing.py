#!/usr/bin/env python3
"""
Timing Data Correction Tool

This tool corrects timing data to align with Quran text by:
1. Removing Isti'aatha (if present) from timing data
2. Aligning timing letters with Quran text (base letters only)
3. Expanding timing to include diacritics from Quran text

The key insight is:
- MAH audio includes Isti'aatha ("أعوذ بالله من الشيطان الرجيم")
- Quran text starts at Basmalah ("بِسْمِ ٱللَّهِ...")
- Timing data has only base letters, Quran text has diacritics
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any

# Arabic diacritics (tashkeel)
DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def is_diacritic(char):
    return char in DIACRITICS

def is_space(char):
    return char.isspace()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_arabic(char):
    """Normalize Arabic characters to their base forms for matching."""
    # Alef variants -> base Alef
    alef_variants = {
        '\u0671': '\u0627',  # Alef wasla -> Alef
        '\u0622': '\u0627',  # Alef with madda above
        '\u0623': '\u0627',  # Alef with hamza above
        '\u0625': '\u0627',  # Alef with hamza below
        '\u0672': '\u0627',  # Alef with wavy hamza above
        '\u0673': '\u0627',  # Alef with wavy hamza below
    }
    return alef_variants.get(char, char)

def extract_base_letters(text):
    """Extract base letters from Arabic text (no diacritics, no spaces)."""
    base = ''
    for char in text:
        if not is_diacritic(char) and not is_space(char):
            base += char
    return base

def find_basmalah_start_in_timing(timing_data):
    """
    Find where Basmalah starts in timing data.
    Basmalah starts with "بسم" (Ba-Sin-Mim without diacritics).
    """
    letters = ''.join(entry.get('char', '') for entry in timing_data)
    
    # Search for "بسم" in the timing letters
    basmalah_start = "بسم"
    idx = letters.find(basmalah_start)
    
    if idx == -1:
        print(f"WARNING: Could not find Basmalah start 'بسم' in timing data")
        # Try alternative (if timing has spaces)
        return 0
    
    return idx

def create_aligned_timing(timing_data, quran_text, basmalah_timing_start_idx):
    """
    Create new timing data that aligns with Quran text.
    
    Strategy:
    - Start timing from basmalah_timing_start_idx
    - For each Quran character:
        - If base letter: assign timing from next timing entry
        - If diacritic: inherit timing from previous base letter
        - If space: skip (no timing needed)
    """
    # Timing data starting from Basmalah
    timing_from_basmalah = timing_data[basmalah_timing_start_idx:]
    
    aligned_timing = []
    timing_idx = 0
    quran_char_idx = 0
    current_word_idx = 0
    last_timing = None
    
    for char in quran_text:
        if is_space(char):
            quran_char_idx += 1
            continue
        
        if is_diacritic(char):
            # Diacritics inherit timing from the previous base letter
            if last_timing:
                aligned_timing.append({
                    'charIdx': len(aligned_timing),
                    'char': char,
                    'start': last_timing['start'],
                    'end': last_timing['end'],
                    'duration': last_timing.get('duration', (last_timing['end'] - last_timing['start']) * 1000),
                    'wordIdx': last_timing.get('wordIdx', current_word_idx)
                })
        else:
            # Base letter - get timing from timing data
            if timing_idx < len(timing_from_basmalah):
                t = timing_from_basmalah[timing_idx]
                
                # Normalize and compare
                timing_char = t.get('char', '')
                quran_base = normalize_arabic(char)
                timing_base = normalize_arabic(timing_char)
                
                # Check if they match
                if quran_base != timing_base:
                    # Try to find the matching timing entry
                    found = False
                    for search_idx in range(timing_idx, min(timing_idx + 3, len(timing_from_basmalah))):
                        if normalize_arabic(timing_from_basmalah[search_idx].get('char', '')) == quran_base:
                            timing_idx = search_idx
                            t = timing_from_basmalah[timing_idx]
                            found = True
                            break
                    
                    if not found:
                        print(f"WARN: No match for Quran char '{char}' at idx {len(aligned_timing)}, timing has '{timing_char}'")
                
                # Get word index from timing
                word_idx = t.get('wordIdx', current_word_idx)
                if word_idx < current_word_idx:
                    word_idx = current_word_idx
                current_word_idx = word_idx
                
                last_timing = {
                    'charIdx': len(aligned_timing),
                    'char': char,
                    'start': t.get('start', 0),
                    'end': t.get('end', 0),
                    'duration': t.get('duration', (t.get('end', 0) - t.get('start', 0)) * 1000),
                    'wordIdx': word_idx
                }
                aligned_timing.append(last_timing)
                timing_idx += 1
            else:
                print(f"WARN: Ran out of timing data at Quran char idx {len(aligned_timing)}, char '{char}'")
        
        quran_char_idx += 1
    
    return aligned_timing

def get_surah_text(verses_path, surah_num):
    """Get concatenated text from surah."""
    all_verses = load_json(verses_path)
    surah_verses = all_verses.get(str(surah_num), [])
    return ''.join(v.get('text', '') for v in surah_verses)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Correct timing data alignment')
    parser.add_argument('surah', type=int, help='Surah number')
    parser.add_argument('--dry-run', action='store_true', help='Print output without saving')
    parser.add_argument('--output', type=str, help='Output file (default: overwrite input)')
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).parent.parent / 'public' / 'data'
    verses_path = base_path / 'verses_v4.json'
    timing_path = base_path / f'letter_timing_{args.surah}.json'
    
    print(f"Processing Surah {args.surah}...")
    
    # Load data
    timing_data = load_json(timing_path)
    quran_text = get_surah_text(verses_path, args.surah)
    
    print(f"Timing entries: {len(timing_data)}")
    print(f"Quran text length: {len(quran_text)}")
    print(f"Quran base letters: {len(extract_base_letters(quran_text))}")
    
    # Find Basmalah start in timing
    basmalah_idx = find_basmalah_start_in_timing(timing_data)
    print(f"Basmalah starts at timing index: {basmalah_idx}")
    
    if basmalah_idx > 0:
        isti_words = ''.join(timing_data[i].get('char', '') for i in range(basmalah_idx))
        print(f"Isti'aatha content (being removed): {isti_words}")
    
    # Create aligned timing
    aligned = create_aligned_timing(timing_data, quran_text, basmalah_idx)
    
    print(f"\nResult: {len(aligned)} aligned timing entries")
    
    # Show preview
    print("\nFirst 10 entries:")
    for i, entry in enumerate(aligned[:10]):
        print(f"  {i}: '{entry['char']}' word={entry['wordIdx']} @ {entry['start']:.3f}s")
    
    # Save
    if args.dry_run:
        print("\n[DRY RUN] Would save to:", args.output or timing_path)
    else:
        output_path = Path(args.output) if args.output else timing_path
        
        # Backup original
        backup_path = timing_path.with_suffix('.backup2.json')
        if not backup_path.exists():
            import shutil
            shutil.copy(timing_path, backup_path)
            print(f"Backed up to: {backup_path}")
        
        save_json(output_path, aligned)
        print(f"Saved corrected timing to: {output_path}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
