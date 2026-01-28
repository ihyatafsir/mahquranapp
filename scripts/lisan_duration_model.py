#!/usr/bin/env python3
"""
Lisan-Based Letter Duration Model (لسان المطابقة)

Based on principles from Lisan al-Arab dictionary:

1. "لزيادة المتحرك على الساكن" (voweled > sukun)
   - Voweled letters (متحرك) are LONGER than silent letters (ساكن)
   
2. مد (Madd) elongation rules from Tajweed:
   - Madd Tabii'i (natural): 2 harakaat (beat units)
   - Madd Wajib/Jaiz: 4-5 harakaat
   - Madd Lazim: 6 harakaat
   
3. Shadda (تشديد): Double the letter = 2x duration

4. Makhraj-based duration adjustments:
   - Throat letters (halq): Longer articulation
   - Lip letters: Quick articulation
   - Emphasized letters: Heavier = longer

This creates a linguistically-grounded duration model that distributes
timing based on Arabic phonetic principles, not just audio guessing.
"""
import json
import numpy as np
from pathlib import Path
import librosa
import torch

from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

# ===========================================================================
# LISAN-BASED PHONETIC CATEGORIES
# ===========================================================================

# Diacritics (harakat) - short vowels attached to letters
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')

# Special marks
SHADDA = '\u0651'  # ّ - Gemination (double letter)
SUKUN = '\u0652'   # ْ - No vowel
FATHA = '\u064E'   # َ - Short 'a'
DAMMA = '\u064F'   # ُ - Short 'u'
KASRA = '\u0650'   # ِ - Short 'i'
TANWIN_FATH = '\u064B'  # ً
TANWIN_DAMM = '\u064C'  # ٌ
TANWIN_KASR = '\u064D'  # ٍ
ALIF_KHANJAR = '\u0670'  # ٰ - Superscript alif (long vowel marker)

# Madd letters (elongation carriers)
MADD_LETTERS = set('اوي')
ALIF_WASLA = 'ٱ'

# Makhraj categories (articulation points)
HALQ_LETTERS = set('ءهعحغخ')      # Throat - deep, longer
SHAFAWI_LETTERS = set('بمو')      # Lips - quick
LISANI_LETTERS = set('تثدذرزسشصضطظلن')  # Tongue - variable
MUFAKHKHAM_LETTERS = set('صضطظخغق')  # Emphatic/heavy - longer
QALQALAH_LETTERS = set('قطبجد')    # Echoing at stop


def get_lisan_duration_weight(char, prev_char=None, next_char=None, in_word_position='middle'):
    """
    Calculate duration weight based on Lisan al-Arab linguistic principles.
    
    Returns a weight multiplier for the base duration unit.
    Base unit = 1.0 (approximately one haraka/beat)
    """
    
    # === DIACRITICS (الحركات) ===
    if char in DIACRITICS:
        if char == SHADDA:
            # تشديد - gemination doubles the consonant
            # The NEXT letter will be doubled, this mark itself is brief
            return 0.3
        elif char == SUKUN:
            # سكون - absence of vowel, very brief
            return 0.15
        elif char in (FATHA, DAMMA, KASRA):
            # Short vowels - "المتحرك" (moving/voweled)
            # These ARE the harakat (beat units)
            return 0.8
        elif char in (TANWIN_FATH, TANWIN_DAMM, TANWIN_KASR):
            # Tanwin (nunation) - slightly longer due to nasal
            return 0.9
        elif char == ALIF_KHANJAR:
            # Superscript alif indicates madd - long vowel
            return 1.5
        else:
            return 0.3
    
    # === MADD LETTERS (حروف المد) ===
    if char in MADD_LETTERS:
        # Check for madd condition: preceded by matching vowel
        if prev_char:
            if char == 'ا' and prev_char == FATHA:
                # Madd with fatha+alif = long 'aa'
                return 3.0
            elif char == 'و' and prev_char == DAMMA:
                # Madd with damma+waw = long 'uu'
                return 3.0
            elif char == 'ي' and prev_char == KASRA:
                # Madd with kasra+ya = long 'ii'
                return 3.0
        
        # Default for madd letters (may be consonantal use)
        return 2.0
    
    # === ALIF WASLA (همزة الوصل) ===
    if char == ALIF_WASLA:
        # Connecting hamza - brief or silent
        return 0.6
    
    # === HALQ LETTERS (حروف الحلق) - Throat ===
    if char in HALQ_LETTERS:
        if char == 'ء':
            # Hamza - glottal stop, brief
            return 0.7
        elif char in 'عح':
            # Ayn and emphatic Ha - deep throat, longer
            return 1.8
        elif char in 'غخ':
            # Ghayn and Kha - fricatives from throat
            return 1.6
        elif char == 'ه':
            # Ha - breathy
            return 1.4
        return 1.5
    
    # === EMPHATIC/HEAVY LETTERS (الحروف المفخمة) ===
    if char in MUFAKHKHAM_LETTERS:
        # Heavy articulation takes more time
        return 1.5
    
    # === SHAFAWI LETTERS (الحروف الشفوية) - Lips ===
    if char in SHAFAWI_LETTERS:
        if char == 'م':
            # Meem has nasal resonance
            return 1.2
        elif char == 'ب':
            # Ba is plosive, quick
            return 1.0
        return 1.1
    
    # === LAM (ل) - Special handling ===
    if char == 'ل':
        # Lam can be heavy (Allah) or light
        # In "الله" the lam is heavy (tafkhim)
        if next_char and next_char == 'ل':
            # First lam in double-lam (لل)
            return 1.3
        return 1.0
    
    # === RA (ر) - Variable ===
    if char == 'ر':
        # Ra is variable - can be heavy or light based on context
        return 1.4
    
    # === NUN (ن) ===
    if char == 'ن':
        # Nasal resonance
        return 1.2
    
    # === SIBILANTS ===
    if char in 'سش':
        # Hissing sounds - slightly longer
        return 1.2
    
    # === INTERDENTALS ===
    if char in 'ثذظ':
        return 1.1
    
    # === DEFAULT ===
    return 1.0


def apply_shadda_doubling(letters_with_weights):
    """
    When shadda appears, double the duration of the PRECEDING consonant.
    This reflects the Arabic principle of تشديد (doubling).
    """
    result = []
    i = 0
    while i < len(letters_with_weights):
        char, weight = letters_with_weights[i]
        
        # Check if next char is shadda
        if i + 1 < len(letters_with_weights):
            next_char, next_weight = letters_with_weights[i + 1]
            if next_char == SHADDA:
                # Double this letter's weight
                weight *= 2.0
        
        result.append((char, weight))
        i += 1
    
    return result


def distribute_timing_lisan(word_text, word_start, word_end):
    """
    Distribute timing across characters using Lisan-based weights.
    """
    if not word_text:
        return []
    
    # Calculate weights for each character
    weights = []
    chars = list(word_text)
    
    for i, char in enumerate(chars):
        prev_char = chars[i-1] if i > 0 else None
        next_char = chars[i+1] if i < len(chars) - 1 else None
        
        position = 'start' if i == 0 else ('end' if i == len(chars) - 1 else 'middle')
        
        weight = get_lisan_duration_weight(char, prev_char, next_char, position)
        weights.append((char, weight))
    
    # Apply shadda doubling
    weights = apply_shadda_doubling(weights)
    
    # Normalize weights
    total_weight = sum(w for _, w in weights)
    if total_weight == 0:
        total_weight = 1
    
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
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    DEVICE = "cpu"
    
    print("=" * 60)
    print("Lisan-Based Letter Duration Model (لسان المطابقة)")
    print("Based on Arabic phonetic principles from Lisan al-Arab")
    print("=" * 60)
    
    # Load text
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get('1', []))
    print(f"\nText: {len(text)} chars")
    
    # Get word-level timing from CTC (accurate word boundaries)
    print("\nLoading CTC aligner for word boundaries...")
    alignment_model, alignment_tokenizer = load_alignment_model(DEVICE, dtype=torch.float32)
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=8)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language='ara')
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"Got {len(word_timestamps)} word alignments")
    
    # Apply Lisan-based distribution within each word
    print("\nApplying Lisan-based phonetic weights...")
    all_timings = []
    
    for wt in word_timestamps:
        word_text = wt['text']
        word_start = wt['start']
        word_end = wt['end']
        
        if not word_text.strip():
            continue
        
        char_timings = distribute_timing_lisan(word_text, word_start, word_end)
        
        for ct in char_timings:
            if not ct['char'].isspace():
                all_timings.append({
                    "idx": len(all_timings),
                    "char": ct['char'],
                    "start": ct['start'],
                    "end": ct['end']
                })
    
    print(f"Total: {len(all_timings)} character timings")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Show preview with weights
    print("\n=== First 20 characters (Lisan weights) ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7} | Weight")
    print("-" * 45)
    
    # Re-run for display with weights
    sample_timings = distribute_timing_lisan(
        word_timestamps[0]['text'] + word_timestamps[1]['text'],
        word_timestamps[0]['start'],
        word_timestamps[1]['end']
    )
    for ct in sample_timings[:20]:
        dur_ms = (ct['end'] - ct['start']) * 1000
        print(f"{ct['char']:^4} | {dur_ms:>6.0f}ms | {ct['start']:>6.3f}s | {ct['weight']:.1f}")


if __name__ == "__main__":
    main()
