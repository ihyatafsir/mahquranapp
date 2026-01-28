#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Diacritic helper (same as previous scripts)
DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
def is_diacritic(char): return char in DIACRITICS
def is_space(char): return char.isspace()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def inject_word_indices(surah_num):
    # Paths
    base_path = Path('public/data')
    timing_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    verses_path = base_path / 'verses_v4.json'
    
    print(f"Processing Surah {surah_num} (Abdul Basit)...")
    
    timing = load_json(timing_path)
    verses_data = load_json(verses_path)
    surah_verses = verses_data.get(str(surah_num), [])
    
    # 1. Map Quran text to words and letter indices
    # We need to know: for simple letter index N (ignoring spaces), what is the word index?
    
    quran_letter_map = [] # maps letter_index -> word_index
    global_word_idx = 0
    current_verse_words = []
    
    # Debug info
    word_debug = []
    
    for verse in surah_verses:
        text = verse['text']
        # Split by spaces to identify words, but keep structure
        # A simple split() might lose multiple spaces, but verses usually single spaced
        raw_words = text.split(' ')
        
        for w in raw_words:
            if not w: continue # skip empty
            
            # For each char in this word
            for char in w:
                if is_space(char): continue
                # We count diacritics as part of the word
                quran_letter_map.append(global_word_idx)
            
            word_debug.append(w)
            global_word_idx += 1
            
    print(f"Total words found in Quran text: {global_word_idx}")
    print(f"Total letter map length: {len(quran_letter_map)}")
    print(f"Timing entries length: {len(timing)}")
    
    if len(quran_letter_map) != len(timing):
        print("❌ CRITICAL: Length mismatch! Cannot inject indices blindly.")
        print(f"Quran letters: {len(quran_letter_map)}")
        print(f"Timing letters: {len(timing)}")
        return False

    # 2. Inject indices
    for i, entry in enumerate(timing):
        entry['wordIdx'] = quran_letter_map[i]
        
    # 3. Save
    save_json(timing_path, timing)
    print("✅ Injected word indices and saved file.")
    return True

if __name__ == "__main__":
    success = inject_word_indices(1)
    sys.exit(0 if success else 1)
