#!/usr/bin/env python3
"""
Batch Fix Timing for All Abdul Basit Surahs

Uses the ORIGINAL timing data (abdul_basit_original) as the source of truth
for audio timestamps. The original data has:
- Base letters only (no diacritics)
- Accurate millisecond timestamps
- May include Isti'aatha at start

This script:
1. Reads original timing data (preserves exact timestamps)
2. Aligns with Quran text (expands for diacritics)
3. Adds wordIdx for proper word highlighting
4. Handles Isti'aatha removal (for surahs that have it)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def is_diacritic(c): return c in DIACRITICS
def is_space(c): return c.isspace()

def normalize_arabic(char):
    variants = {
        '\u0671': '\u0627',
        '\u0622': '\u0627',
        '\u0623': '\u0627',
        '\u0625': '\u0627',
    }
    return variants.get(char, char)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Remove BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        return json.loads(content)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_quran_text(verses_path, surah_num):
    all_verses = load_json(verses_path)
    verses = all_verses.get(str(surah_num), [])
    return ''.join(v['text'] for v in verses)

def get_quran_words(verses_path, surah_num):
    all_verses = load_json(verses_path)
    verses = all_verses.get(str(surah_num), [])
    words = []
    for v in verses:
        for w in v['text'].split():
            if w.strip():
                words.append(w)
    return words

def find_quran_start_in_timing(timing_data, quran_text):
    """
    Find where Quran text starts in timing data.
    For most surahs, timing starts with Basmalah (بسم).
    For Surah 1, timing may include Isti'aatha first.
    For Surah 9, there's no Basmalah.
    """
    quran_base = [c for c in quran_text if not is_diacritic(c) and not is_space(c)]
    if not quran_base:
        return 0
    
    first_quran_char = normalize_arabic(quran_base[0])
    
    # Search for matching start
    for i, entry in enumerate(timing_data):
        timing_char = normalize_arabic(entry.get('char', ''))
        if timing_char == first_quran_char:
            # Verify next few chars match
            match_count = 0
            for j in range(min(5, len(quran_base), len(timing_data) - i)):
                q = normalize_arabic(quran_base[j])
                t = normalize_arabic(timing_data[i+j].get('char', ''))
                if q == t:
                    match_count += 1
            if match_count >= 3:
                return i
    
    return 0

def process_surah(surah_num, base_path, verses_path, dry_run=False):
    """Process a single surah."""
    original_path = base_path / f'abdul_basit_original/letter_timing_{surah_num}.json'
    output_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    
    if not original_path.exists():
        return None, f"Original file not found: {original_path}"
    
    original = load_json(original_path)
    
    # Filter out BOM entries
    original = [e for e in original if e.get('char', '') != '\ufeff']
    
    quran_text = get_quran_text(verses_path, surah_num)
    quran_words = get_quran_words(verses_path, surah_num)
    
    if not quran_text:
        return None, f"No Quran text found for surah {surah_num}"
    
    # Find where Quran text starts in timing
    start_idx = find_quran_start_in_timing(original, quran_text)
    timing_from_start = original[start_idx:]
    
    # Build word index map
    word_map = []
    word_idx = 0
    for w in quran_words:
        for c in w:
            if not is_space(c):
                word_map.append(word_idx)
        word_idx += 1
    
    # Align timing with Quran text
    aligned = []
    timing_idx = 0
    char_global_idx = 0
    last_timing = None
    
    for c in quran_text:
        if is_space(c):
            continue
        
        if is_diacritic(c):
            if last_timing:
                aligned.append({
                    'idx': len(aligned),
                    'char': c,
                    'start': last_timing['start'],
                    'end': last_timing['end'],
                    'duration': last_timing['duration'],
                    'wordIdx': word_map[char_global_idx] if char_global_idx < len(word_map) else 0
                })
            char_global_idx += 1
        else:
            if timing_idx < len(timing_from_start):
                t = timing_from_start[timing_idx]
                
                last_timing = {
                    'start': t.get('start', 0),
                    'end': t.get('end', 0),
                    'duration': t.get('duration', t.get('end', 0) - t.get('start', 0))
                }
                
                aligned.append({
                    'idx': len(aligned),
                    'char': c,
                    'start': last_timing['start'],
                    'end': last_timing['end'],
                    'duration': last_timing['duration'],
                    'wordIdx': word_map[char_global_idx] if char_global_idx < len(word_map) else 0
                })
                timing_idx += 1
            
            char_global_idx += 1
    
    if not dry_run:
        save_json(output_path, aligned)
    
    return {
        'surah': surah_num,
        'original_entries': len(original),
        'timing_start_idx': start_idx,
        'aligned_entries': len(aligned),
        'word_count': len(quran_words)
    }, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch fix Abdul Basit timing')
    parser.add_argument('--surah', type=int, help='Process single surah')
    parser.add_argument('--dry-run', action='store_true', help='Preview without saving')
    args = parser.parse_args()
    
    base_path = Path('public/data')
    verses_path = base_path / 'verses_v4.json'
    
    if args.surah:
        surahs = [args.surah]
    else:
        surahs = range(1, 115)
    
    results = []
    errors = []
    
    for surah_num in surahs:
        result, error = process_surah(surah_num, base_path, verses_path, args.dry_run)
        if error:
            errors.append(f"Surah {surah_num}: {error}")
        elif result:
            results.append(result)
            print(f"Surah {surah_num}: {result['aligned_entries']} entries, {result['word_count']} words")
    
    print(f"\nProcessed {len(results)} surahs successfully")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were modified")

if __name__ == '__main__':
    main()
