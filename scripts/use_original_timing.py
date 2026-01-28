#!/usr/bin/env python3
"""
Use ORIGINAL Abdul Basit timing data (base letters only) and properly align
with full Quran text (with diacritics).

Strategy:
1. Load original timing data (176 entries for Surah 1)
2. Skip Isti'aatha entries (before Basmalah)
3. For each Quran text character:
   - If base letter: assign timing from original data
   - If diacritic: inherit timing from previous base letter
4. Inject proper wordIdx based on Quran word structure
"""

import json
import sys
from pathlib import Path

DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
def is_diacritic(c): return c in DIACRITICS
def is_space(c): return c.isspace()

def normalize_arabic(char):
    """Normalize Arabic chars for matching."""
    variants = {
        '\u0671': '\u0627',  # Alef wasla -> Alef
        '\u0622': '\u0627',  # Alef madda
        '\u0623': '\u0627',  # Alef hamza above
        '\u0625': '\u0627',  # Alef hamza below
    }
    return variants.get(char, char)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_basmalah_start(timing_data):
    """Find where بسم starts in timing data."""
    for i, e in enumerate(timing_data):
        if i+1 < len(timing_data) and e['char'] == 'ب' and timing_data[i+1]['char'] == 'س':
            return i
    return 0

def get_quran_text(verses_path, surah_num):
    all_verses = load_json(verses_path)
    verses = all_verses.get(str(surah_num), [])
    return ''.join(v['text'] for v in verses)

def get_quran_words(verses_path, surah_num):
    """Get list of words for wordIdx assignment."""
    all_verses = load_json(verses_path)
    verses = all_verses.get(str(surah_num), [])
    words = []
    for v in verses:
        for w in v['text'].split():
            if w.strip():
                words.append(w)
    return words

def build_aligned_timing(surah_num):
    base_path = Path('public/data')
    original_path = base_path / f'abdul_basit_original/letter_timing_{surah_num}.json'
    verses_path = base_path / 'verses_v4.json'
    output_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    
    print(f"Processing Surah {surah_num}...")
    
    original = load_json(original_path)
    quran_text = get_quran_text(verses_path, surah_num)
    quran_words = get_quran_words(verses_path, surah_num)
    
    # Find Basmalah start in original timing
    basmalah_idx = find_basmalah_start(original)
    print(f"Basmalah starts at original index {basmalah_idx} (time {original[basmalah_idx]['start']}ms)")
    
    # Timing data from Basmalah onwards
    timing_from_basmalah = original[basmalah_idx:]
    print(f"Timing entries from Basmalah: {len(timing_from_basmalah)}")
    
    # Count base letters in Quran text
    quran_base_letters = [c for c in quran_text if not is_diacritic(c) and not is_space(c)]
    print(f"Quran base letters: {len(quran_base_letters)}")
    
    # Build word index map (which word each non-space char belongs to)
    word_map = []
    word_idx = 0
    for w in quran_words:
        for c in w:
            if not is_space(c):
                word_map.append(word_idx)
        word_idx += 1
    
    # Build aligned timing
    aligned = []
    timing_idx = 0
    char_global_idx = 0  # Index into non-space chars (for word_map)
    last_timing = None
    
    for c in quran_text:
        if is_space(c):
            continue
        
        if is_diacritic(c):
            # Diacritics inherit timing from previous base letter
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
            # Base letter - get timing from original data
            if timing_idx < len(timing_from_basmalah):
                t = timing_from_basmalah[timing_idx]
                
                # Verify match (allow normalization differences)
                orig_char = t['char']
                quran_norm = normalize_arabic(c)
                orig_norm = normalize_arabic(orig_char)
                
                if quran_norm != orig_norm:
                    # Try to resync by searching ahead
                    found = False
                    for search in range(timing_idx, min(timing_idx+5, len(timing_from_basmalah))):
                        if normalize_arabic(timing_from_basmalah[search]['char']) == quran_norm:
                            timing_idx = search
                            t = timing_from_basmalah[timing_idx]
                            found = True
                            break
                    if not found:
                        print(f"WARN: Mismatch at aligned idx {len(aligned)}: Quran '{c}' vs Timing '{orig_char}'")
                
                last_timing = {
                    'start': t['start'],
                    'end': t['end'],
                    'duration': t.get('duration', t['end'] - t['start'])
                }
                
                aligned.append({
                    'idx': len(aligned),
                    'char': c,
                    'start': t['start'],
                    'end': t['end'],
                    'duration': last_timing['duration'],
                    'wordIdx': word_map[char_global_idx] if char_global_idx < len(word_map) else 0
                })
                timing_idx += 1
            else:
                print(f"WARN: Ran out of timing at aligned idx {len(aligned)}, char '{c}'")
            
            char_global_idx += 1
    
    print(f"Final aligned entries: {len(aligned)}")
    
    # Save
    save_json(output_path, aligned)
    print(f"Saved to {output_path}")
    
    return aligned

if __name__ == "__main__":
    build_aligned_timing(1)
