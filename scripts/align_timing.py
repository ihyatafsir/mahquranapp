#!/usr/bin/env python3
"""
Align WhisperX letter-level timing with actual Quran text.
Preserves original letter timestamps while using correct Quran characters.
"""

import json
import os
import re
from pathlib import Path

DATA_DIR = Path('/home/absolut7/Documents/26apps/MahQuranApp/public/data')
OUTPUT_DIR = DATA_DIR

PREFIX_PATTERNS = [
    ['أعوذ', 'بالله', 'من', 'الشيطان', 'الرجيم'],
    ['اعوذ', 'بالله', 'من', 'الشيطان', 'الرجيم'],
    ['بسم', 'الله', 'الرحمن', 'الرحيم'],
    ['بسم', 'الله', 'الرحمان', 'الرحيم'],
]

def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics and normalize."""
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')
    text = diacritics.sub('', text)
    text = re.sub(r'[إأآٱا]', 'ا', text)
    return text

def load_quran_text(surah_num: int) -> list[dict]:
    with open(DATA_DIR / 'verses_v4.json') as f:
        return json.load(f).get(str(surah_num), [])

def load_original_timing(surah_num: int) -> list[dict]:
    """Load original letter timing from backup."""
    backup_path = DATA_DIR / f'letter_timing_{surah_num}.backup.json'
    if backup_path.exists():
        with open(backup_path) as f:
            return json.load(f)
    
    timing_path = DATA_DIR / f'letter_timing_{surah_num}.json'
    if timing_path.exists():
        with open(timing_path) as f:
            return json.load(f)
    return []

def get_timing_words(timing: list[dict]) -> dict:
    """Group letters by wordIdx, preserving letter-level timing."""
    words = {}
    for t in timing:
        wid = t['wordIdx']
        if wid not in words:
            words[wid] = []
        words[wid].append(t)
    return words

def get_quran_words(verses: list[dict]) -> list[str]:
    words = []
    for verse in verses:
        words.extend(verse.get('text', '').split())
    return words

def find_prefix_length(timing_words: dict, quran_first_word: str) -> int:
    """Find how many words in timing are prefix."""
    quran_first_clean = strip_diacritics(quran_first_word)
    
    for wid in sorted(timing_words.keys())[:15]:
        word_chars = ''.join([t['char'] for t in timing_words[wid]])
        timing_clean = strip_diacritics(word_chars)
        if timing_clean == quran_first_clean or quran_first_clean.startswith(timing_clean):
            return wid
    
    # Check for known patterns
    timing_text = ' '.join([strip_diacritics(''.join([t['char'] for t in timing_words[i]])) 
                           for i in sorted(timing_words.keys())[:10]])
    for pattern in PREFIX_PATTERNS:
        pattern_text = ' '.join([strip_diacritics(p) for p in pattern])
        if pattern_text in timing_text:
            return len(pattern)
    
    return 0

def align_letters_to_quran(timing_words: dict, quran_words: list[str], prefix_len: int) -> list[dict]:
    """
    Map letter-level timing to Quran characters.
    Preserves original timing per letter while using correct Quran text.
    """
    aligned = []
    
    # Get timing wordIds after prefix
    timing_wids = sorted([w for w in timing_words.keys() if w >= prefix_len])
    
    # Check if timing is in ms
    first_letter = timing_words[timing_wids[0]][0] if timing_wids and timing_words[timing_wids[0]] else None
    is_ms = first_letter and first_letter['start'] > 100
    
    for i, quran_word in enumerate(quran_words):
        if i >= len(timing_wids):
            print(f"    Warning: Ran out of timing at word {i}")
            break
        
        timing_wid = timing_wids[i]
        timing_letters = timing_words[timing_wid]
        
        # Get base letters from both (without diacritics)
        quran_base = list(strip_diacritics(quran_word))
        timing_base = [strip_diacritics(t['char']) for t in timing_letters]
        
        # Map timing to Quran letters
        t_idx = 0  # Index into timing letters
        q_base_idx = 0  # Index into quran base letters
        
        for q_idx, q_char in enumerate(quran_word):
            q_is_base = bool(strip_diacritics(q_char))
            
            if q_is_base and t_idx < len(timing_letters):
                # Use timing from original letter
                t = timing_letters[t_idx]
                start = t['start'] / 1000 if is_ms else t['start']
                end = t['end'] / 1000 if is_ms else t['end']
                duration = (end - start) * 1000
                t_idx += 1
                q_base_idx += 1
            elif aligned:
                # Diacritic - use previous letter's timing
                start = aligned[-1]['start']
                end = aligned[-1]['end']
                duration = aligned[-1]['duration']
            else:
                # First char is diacritic (shouldn't happen)
                t = timing_letters[0] if timing_letters else {'start': 0, 'end': 0}
                start = t['start'] / 1000 if is_ms else t['start']
                end = t['end'] / 1000 if is_ms else t['end']
                duration = (end - start) * 1000
            
            aligned.append({
                'charIdx': q_idx,
                'char': q_char,
                'start': round(start, 3),
                'end': round(end, 3),
                'duration': round(duration, 1),
                'wordIdx': i
            })
    
    return aligned

def process_surah(surah_num: int) -> bool:
    print(f"\n{'='*50}")
    print(f"Processing Surah {surah_num}")
    print('='*50)
    
    verses = load_quran_text(surah_num)
    if not verses:
        print(f"  No verses found")
        return False
    
    timing = load_original_timing(surah_num)
    if not timing:
        print(f"  No timing data found")
        return False
    
    timing_words = get_timing_words(timing)
    quran_words = get_quran_words(verses)
    
    print(f"  Timing words: {len(timing_words)}")
    print(f"  Quran words: {len(quran_words)}")
    print(f"  Original letters: {len(timing)}")
    
    prefix_len = find_prefix_length(timing_words, quran_words[0] if quran_words else '')
    print(f"  Prefix to skip: {prefix_len} words")
    
    if prefix_len > 0:
        skipped = [strip_diacritics(''.join([t['char'] for t in timing_words[w]])) 
                   for w in sorted(timing_words.keys())[:prefix_len]]
        print(f"  Skipping: {skipped}")
    
    aligned = align_letters_to_quran(timing_words, quran_words, prefix_len)
    print(f"  Aligned letters: {len(aligned)}")
    
    # Show sample
    print(f"  Sample (first 3 words):")
    for wid in range(min(3, max([a['wordIdx'] for a in aligned]) + 1)):
        word_letters = [a for a in aligned if a['wordIdx'] == wid]
        word_text = ''.join([a['char'] for a in word_letters])
        first_t = word_letters[0]['start'] if word_letters else 0
        last_t = word_letters[-1]['end'] if word_letters else 0
        print(f"    Word {wid}: '{word_text}' ({first_t:.3f}s - {last_t:.3f}s)")
    
    output_path = OUTPUT_DIR / f'letter_timing_{surah_num}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {output_path.name}")
    
    return True

def main():
    surahs = [1, 18, 36, 47, 53, 55, 56, 67, 71, 75, 80, 82, 85, 87, 89, 90, 91, 92, 93, 109, 112, 113, 114]
    
    print("Aligning letter-level timing with Quran text")
    print("Preserving original letter timestamps")
    print("=" * 60)
    
    success = 0
    for surah in surahs:
        if process_surah(surah):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {success}/{len(surahs)} surahs")

if __name__ == '__main__':
    main()
