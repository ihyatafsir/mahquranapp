#!/usr/bin/env python3
"""
Smart Word-Based Alignment

Uses word boundaries as checkpoints and context matching to align:
- Groups timing by words (using timing gaps)
- Aligns timing words to Quran words
- Uses surrounding context to handle character mismatches
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def is_diacritic(c): return c in DIACRITICS
def is_space(c): return c.isspace()

def normalize_char(c):
    """Normalize Arabic character variants for matching."""
    return {'ٱ': 'ا', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ي', 'ى': 'ي'}.get(c, c)

def get_base_letters(text):
    """Extract base letters only (no diacritics, no spaces)."""
    return [c for c in text if not is_diacritic(c) and not is_space(c)]

def group_timing_into_words(timing_data, gap_threshold_ms=15):
    """Group timing entries into words based on timing gaps."""
    words = []
    current_word = []
    
    for i, entry in enumerate(timing_data):
        if i > 0:
            prev_end = timing_data[i-1].get('end', 0)
            curr_start = entry.get('start', 0)
            gap = curr_start - prev_end
            if gap > gap_threshold_ms:
                if current_word:
                    words.append(current_word)
                current_word = []
        current_word.append(entry)
    
    if current_word:
        words.append(current_word)
    
    return words

def get_quran_words(verses):
    """Get list of words from Quran verses."""
    words = []
    for verse in verses:
        text = verse.get('text', '')
        for word in text.split():
            if word.strip():
                words.append({
                    'text': word,
                    'base_letters': get_base_letters(word)
                })
    return words

def word_similarity(timing_word_entries, quran_word):
    """Calculate similarity between timing word and Quran word."""
    timing_chars = ''.join([normalize_char(e['char']) for e in timing_word_entries])
    quran_chars = ''.join([normalize_char(c) for c in quran_word['base_letters']])
    return SequenceMatcher(None, timing_chars, quran_chars).ratio()

def align_words(timing_words, quran_words):
    """
    Align timing words to Quran words using dynamic programming.
    Returns list of (timing_word_idx, quran_word_idx) pairs.
    """
    alignments = []
    t_idx = 0
    q_idx = 0
    
    while t_idx < len(timing_words) and q_idx < len(quran_words):
        # Calculate similarity with current Quran word
        sim = word_similarity(timing_words[t_idx], quran_words[q_idx])
        
        if sim > 0.5:
            # Good match - align them
            alignments.append((t_idx, q_idx))
            t_idx += 1
            q_idx += 1
        elif sim < 0.3:
            # Poor match - try to find better alignment
            # Check if next timing word matches better
            if t_idx + 1 < len(timing_words):
                next_sim = word_similarity(timing_words[t_idx + 1], quran_words[q_idx])
                if next_sim > sim:
                    t_idx += 1  # Skip this timing word
                    continue
            # Check if next Quran word matches better
            if q_idx + 1 < len(quran_words):
                next_sim = word_similarity(timing_words[t_idx], quran_words[q_idx + 1])
                if next_sim > sim:
                    q_idx += 1  # Skip this Quran word
                    continue
            # Neither helps - just align and move on
            alignments.append((t_idx, q_idx))
            t_idx += 1
            q_idx += 1
        else:
            # Moderate match - accept it
            alignments.append((t_idx, q_idx))
            t_idx += 1
            q_idx += 1
    
    return alignments

def build_letter_mapping(timing_words, quran_words, alignments):
    """
    Build letter-level mapping from word alignments.
    Each Quran base letter maps to a timing entry's start/end times.
    """
    mapping = []  # List of {quran_letter_idx, timing_entry}
    
    quran_letter_idx = 0
    
    for t_word_idx, q_word_idx in alignments:
        timing_entries = timing_words[t_word_idx]
        quran_word = quran_words[q_word_idx]
        quran_base = quran_word['base_letters']
        
        # Align letters within word
        for i, q_char in enumerate(quran_base):
            if i < len(timing_entries):
                mapping.append({
                    'quran_idx': quran_letter_idx,
                    'timing': timing_entries[i]
                })
            elif timing_entries:
                # More Quran letters than timing - use last timing
                mapping.append({
                    'quran_idx': quran_letter_idx,
                    'timing': timing_entries[-1]
                })
            quran_letter_idx += 1
        
        # Handle case where timing has more entries than Quran letters
        # (skip extra timing entries - they're probably errors)
    
    return mapping

def process_surah(surah_num, base_path):
    """Process a single surah with word-based alignment."""
    original_path = base_path / f'abdul_basit_original/letter_timing_{surah_num}.json'
    verses_path = base_path / 'verses_v4.json'
    output_path = base_path / f'abdul_basit/letter_timing_{surah_num}.json'
    
    if not original_path.exists():
        return None, f"File not found: {original_path}"
    
    # Load data
    timing_data = json.load(open(original_path, 'r', encoding='utf-8'))
    timing_data = [e for e in timing_data if e.get('char', '') != '\ufeff']
    
    all_verses = json.load(open(verses_path, 'r', encoding='utf-8'))
    verses = all_verses.get(str(surah_num), [])
    
    if not verses:
        return None, f"No verses for surah {surah_num}"
    
    # Group timing into words
    timing_words = group_timing_into_words(timing_data)
    quran_words = get_quran_words(verses)
    
    # Align words
    alignments = align_words(timing_words, quran_words)
    
    # Build letter mapping
    letter_mapping = build_letter_mapping(timing_words, quran_words, alignments)
    
    # Create output timing data with correct indices
    output = []
    for i, entry in enumerate(letter_mapping):
        t = entry['timing']
        output.append({
            'idx': i,
            'char': t.get('char', ''),
            'start': t.get('start', 0),
            'end': t.get('end', 0),
            'duration': t.get('duration', 0),
            'wordIdx': 0  # Will be set below
        })
    
    # Assign wordIdx based on Quran words
    word_idx = 0
    letter_in_word = 0
    for i, entry in enumerate(output):
        if word_idx < len(quran_words):
            entry['wordIdx'] = word_idx
            letter_in_word += 1
            if letter_in_word >= len(quran_words[word_idx]['base_letters']):
                word_idx += 1
                letter_in_word = 0
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return {
        'surah': surah_num,
        'timing_words': len(timing_words),
        'quran_words': len(quran_words),
        'alignments': len(alignments),
        'output_entries': len(output)
    }, None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--surah', type=int, help='Single surah')
    args = parser.parse_args()
    
    base_path = Path('public/data')
    
    if args.surah:
        result, error = process_surah(args.surah, base_path)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Result: {result}")
    else:
        for surah_num in range(1, 115):
            result, error = process_surah(surah_num, base_path)
            if error:
                print(f"Surah {surah_num}: ERROR - {error}")
            else:
                print(f"Surah {surah_num}: {result['output_entries']} entries, "
                      f"{result['alignments']}/{result['quran_words']} words aligned")

if __name__ == '__main__':
    main()
