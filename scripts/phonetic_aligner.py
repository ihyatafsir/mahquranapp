#!/usr/bin/env python3
"""
Custom Phonetic Alignment Engine

Uses Arabic phonetic rules (tajweed) to distribute letter timing within words.
This is a hybrid approach:
1. Get word-level timing from CTC aligner (accurate word boundaries)
2. Distribute letter timing within words using phonetic weights

Phonetic Weights:
- Madd letters (ا, و, ي after fatha/damma/kasra): 3x duration
- Shadda (ّ): 2x duration (gemination)
- Base consonants: 1x duration
- Diacritics (harakat): 0.3x duration (attached to base letter)
"""
import json
import torch
from pathlib import Path
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

# Phonetic weight mapping based on tajweed rules
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
MADD_LETTERS = set('اوي')  # Alif, Waw, Ya
SHADDA = '\u0651'  # ّ
SUKUN = '\u0652'  # ْ

def get_phonetic_weight(char, prev_char=None):
    """
    Calculate phonetic duration weight for a character based on tajweed.
    """
    if char in DIACRITICS:
        if char == SHADDA:
            return 2.0  # Gemination doubles the letter
        elif char == SUKUN:
            return 0.3  # Sukun is short
        else:
            return 0.4  # Regular diacritics are short
    
    # Madd letters get extended duration when preceded by matching vowel
    if char in MADD_LETTERS:
        # Simplified: just give madd letters more weight
        return 2.5
    
    # Regular consonants
    return 1.0

def distribute_within_word(word_text, word_start, word_end):
    """
    Distribute timing across characters in a word using phonetic weights.
    """
    if not word_text:
        return []
    
    # Calculate weights for each character
    weights = []
    for i, char in enumerate(word_text):
        prev_char = word_text[i-1] if i > 0 else None
        weight = get_phonetic_weight(char, prev_char)
        weights.append((char, weight))
    
    # Normalize weights to sum to 1
    total_weight = sum(w for _, w in weights)
    if total_weight == 0:
        total_weight = 1
    
    # Calculate duration per unit weight
    word_duration = word_end - word_start
    
    # Generate timings
    timings = []
    current_time = word_start
    
    for char, weight in weights:
        char_duration = (weight / total_weight) * word_duration
        timings.append({
            "char": char,
            "start": current_time,
            "end": current_time + char_duration,
            "weight": weight
        })
        current_time += char_duration
    
    return timings

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    DEVICE = "cpu"
    
    print("=== Custom Phonetic Alignment Engine ===")
    
    # 1. Load text
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get('1', []))
    print(f"Text length: {len(text)} chars")
    
    # 2. Get word-level timing from CTC aligner
    print("Loading CTC aligner for word boundaries...")
    alignment_model, alignment_tokenizer = load_alignment_model(DEVICE, dtype=torch.float32)
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=8)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language='ara')
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"Got {len(word_timestamps)} word alignments")
    
    # 3. Distribute within each word using phonetic weights
    print("Applying phonetic weights...")
    all_char_timings = []
    
    for wt in word_timestamps:
        word_text = wt['text']
        word_start = wt['start']
        word_end = wt['end']
        
        # Skip empty words
        if not word_text.strip():
            continue
        
        # Distribute timing within this word
        char_timings = distribute_within_word(word_text, word_start, word_end)
        
        for ct in char_timings:
            if not ct['char'].isspace():
                all_char_timings.append({
                    "idx": len(all_char_timings),
                    "char": ct['char'],
                    "start": ct['start'],
                    "end": ct['end']
                })
    
    print(f"Total chars: {len(all_char_timings)}")
    
    # 4. Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_char_timings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Print first 15 for verification
    print("\nFirst 15 characters (with phonetic weights):")
    for ct in all_char_timings[:15]:
        dur = ct['end'] - ct['start']
        print(f"  {ct['idx']:3d}: '{ct['char']}' @ {ct['start']:.3f}s - {ct['end']:.3f}s ({dur*1000:.0f}ms)")

if __name__ == "__main__":
    main()
