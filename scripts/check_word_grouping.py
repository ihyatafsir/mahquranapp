#!/usr/bin/env python3
"""
Test tool to simulate App.tsx word grouping logic and compare with Quran word counts.
"""

import json
import sys
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_letters_into_words_simulated(timing_data):
    """
    Simulate the word grouping logic from App.tsx (groupLettersIntoWords).
    """
    words = []
    current_word = None
    auto_word_idx = 0
    
    has_word_idx = len(timing_data) > 0 and 'wordIdx' in timing_data[0]
    
    print(f"Data has wordIdx: {has_word_idx}")
    
    for i, letter in enumerate(timing_data):
        effective_word_idx = letter.get('wordIdx')
        
        # App.tsx logic for auto-detection
        if not has_word_idx:
            prev_letter = timing_data[i-1] if i > 0 else None
            # "start" and "end" are in seconds in App.tsx after normalization
            # Here they are likely in milliseconds or seconds depending on file.
            # Convert to seconds for comparison threshold
            
            l_start = letter['start']
            l_end = letter['end']
            
            # Check if likely milliseconds
            if l_start > 100: 
                l_start /= 1000.0
                l_end /= 1000.0
                
            prev_end = 0
            if prev_letter:
                prev_end = prev_letter['end']
                if prev_end > 100: prev_end /= 1000.0
            
            gap = l_start - prev_end
            if prev_letter and gap > 0.05: # 50ms gap
                auto_word_idx += 1
            
            effective_word_idx = auto_word_idx

        if current_word is None or effective_word_idx != current_word['wordIdx']:
            if current_word:
                words.append(current_word)
            current_word = {
                'wordIdx': effective_word_idx,
                'text': '',
                'char_count': 0
            }
        
        current_word['text'] += letter['char']
        current_word['char_count'] += 1
        
    if current_word:
        words.append(current_word)
        
    return words

def main():
    surah_num = 1
    base_path = Path('public/data')
    timing_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    verses_path = base_path / 'verses_v4.json'
    
    print(f"Analyzing Surah {surah_num} (Abdul Basit)...")
    
    timing = load_json(timing_path)
    verses_data = load_json(verses_path)
    surah_verses = verses_data.get(str(surah_num), [])
    
    # 1. Expected Words from Quran Text
    expected_words = []
    for v in surah_verses:
        v_words = v['text'].strip().split(' ')
        expected_words.extend([w for w in v_words if w]) # Filter empty
        
    print(f"\nExpected Word Count (Quran): {len(expected_words)}")
    print(f"Expected words (first 5): {expected_words[:5]}")
    
    # 2. Simulated Detected Words
    detected_words = group_letters_into_words_simulated(timing)
    print(f"\nDetected Word Count (Timing Auto-detect): {len(detected_words)}")
    print(f"Detected words (first 5): {[w['text'] for w in detected_words[:5]]}")
    
    if len(expected_words) != len(detected_words):
        print(f"\n❌ MISMATCH! Expected {len(expected_words)} words, found {len(detected_words)}.")
        print("This confirms why highlighting is off.")
    else:
        print("\n✅ Count matches! Layout might be okay, checking content...")
        for i in range(min(len(expected_words), len(detected_words))):
            # Normalize for simple check
            e = ''.join(c for c in expected_words[i] if not c.isspace())
            d = detected_words[i]['text']
            # allow some fuzziness due to diacritics
            if len(e) != len(d): # Very rough check
                 print(f"  Warning at word {i}: Quran '{e}' vs Timing '{d}'")

if __name__ == "__main__":
    main()
