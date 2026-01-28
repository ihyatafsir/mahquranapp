#!/usr/bin/env python3
"""
Simple Offset-Based Alignment

Approach:
1. Copy original timing data as-is (preserve all timestamps)
2. Find where the Quran text starts in the timing (skip Isti'aatha if present)
3. Map timing indices to Quran indices with proper offset
4. Store the offset so App.tsx can use it
"""

import json
from pathlib import Path

DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def normalize_char(c):
    return {'ٱ': 'ا', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا'}.get(c, c)

def get_base_letters(text):
    return [c for c in text if c not in DIACRITICS and not c.isspace()]

def find_quran_start_offset(timing_data, quran_base_letters):
    """Find where Quran text starts in timing data."""
    if not timing_data or not quran_base_letters:
        return 0
    
    # Normalize first few Quran letters for matching
    quran_start = ''.join([normalize_char(c) for c in quran_base_letters[:5]])
    
    # Search in timing data
    for offset in range(min(50, len(timing_data))):
        timing_chars = ''.join([normalize_char(timing_data[i]['char']) 
                                for i in range(offset, min(offset+5, len(timing_data)))])
        if timing_chars == quran_start:
            return offset
    
    return 0  # No offset found, assume aligned

def process_surah(surah_num, base_path):
    original_path = base_path / f'abdul_basit_original/letter_timing_{surah_num}.json'
    verses_path = base_path / 'verses_v4.json'
    output_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    
    if not original_path.exists():
        return None, f"Not found: {original_path}"
    
    # Load original timing
    timing = json.load(open(original_path, 'r', encoding='utf-8'))
    timing = [t for t in timing if t.get('char', '') != '\ufeff']
    
    # Load Quran text
    all_verses = json.load(open(verses_path, 'r', encoding='utf-8'))
    verses = all_verses.get(str(surah_num), [])
    quran_text = ''.join([v['text'] for v in verses])
    quran_base = get_base_letters(quran_text)
    
    # Find offset
    offset = find_quran_start_offset(timing, quran_base)
    
    # Create output: original timing from offset, with sequential indices
    output = []
    for i, t in enumerate(timing[offset:]):
        if i >= len(quran_base):
            break  # Stop when we've covered all Quran letters
        output.append({
            'idx': i,
            'char': t.get('char', ''),
            'start': t.get('start', 0),
            'end': t.get('end', 0),
            'duration': t.get('duration', 0),
            'ayah': t.get('ayah', 1)
        })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return {
        'surah': surah_num,
        'offset': offset,
        'timing_entries': len(timing),
        'output_entries': len(output),
        'quran_letters': len(quran_base)
    }, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--surah', type=int)
    args = parser.parse_args()
    
    base_path = Path('public/data')
    
    surahs = [args.surah] if args.surah else range(1, 115)
    
    for surah_num in surahs:
        result, error = process_surah(surah_num, base_path)
        if error:
            print(f"Surah {surah_num}: ERROR - {error}")
        else:
            status = "✅" if result['output_entries'] >= result['quran_letters'] * 0.9 else "⚠️"
            print(f"Surah {surah_num}: {status} offset={result['offset']}, "
                  f"output={result['output_entries']}/{result['quran_letters']}")

if __name__ == '__main__':
    main()
