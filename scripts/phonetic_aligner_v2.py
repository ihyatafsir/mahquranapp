#!/usr/bin/env python3
"""
Enhanced Phonetic Alignment Engine v2

Uses Arabic makhraj (articulation points) and sifat (characteristics) patterns
from classical Arabic linguistics (inspired by Lisan al-Arab) to distribute
letter timing with precision.

Letter Categories by Makhraj (articulation point):
1. حلق (Throat): ء ه ع ح غ خ - Deep articulation, naturally longer
2. لسان (Tongue): Most letters - Variable duration
3. شفوي (Lips): ب م و - Quick articulation
4. خيشوم (Nasal): ن م (when nasal) - Medium duration

Letter Characteristics (Sifat):
1. مفخمة (Emphatic): ص ض ط ظ خ غ ق - Heavy, longer duration
2. مرققة (Light): Non-emphatic - Shorter duration
3. شديدة (Plosive): Sudden release - Shorter
4. رخوة (Fricative): Continuous - Longer
5. مد (Elongation): ا و ي - Context-dependent elongation

Tajweed Rules:
- Shadda (ّ): Gemination = 2x duration
- Madd Asli: 2 harakaat
- Madd Far'i: 2-6 harakaat depending on context
- Sukun (ْ): Very short
"""
import json
import torch
from pathlib import Path
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

# =============================================================================
# ARABIC LETTER CLASSIFICATION (Based on Makhraj/Sifat)
# =============================================================================

# Diacritics (harakat) - very short
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')

# Special marks
SHADDA = '\u0651'  # ّ - Gemination
SUKUN = '\u0652'   # ْ - No vowel
FATHA = '\u064E'   # َ
DAMMA = '\u064F'   # ُ  
KASRA = '\u0650'   # ِ
ALIF_KHANJARIYYA = '\u0670'  # ٰ - Superscript alif

# Madd letters (elongation vowels)
MADD_LETTERS = set('اوي')
ALIF_WASLA = 'ٱ'  # Alif with wasla - doesn't elongate

# Letters by Makhraj (articulation point)
# Throat letters (حلق) - deep articulation = naturally longer
HALQ_LETTERS = set('ءهعحغخ')

# Emphatic/Heavy letters (مفخمة) - require more effort = longer
MUFAKHKHAM_LETTERS = set('صضطظخغق')

# Qalqalah letters (قلقلة) - echoing/bouncing = slightly longer
QALQALAH_LETTERS = set('قطبجد')

# Lip letters (شفوية) - quick articulation = shorter
SHAFAWI_LETTERS = set('بمو')

# Light/soft letters - shorter duration
MURAKKAK_LETTERS = set('ثذسشفكلنهيت')

# Definite article ال
LAM = 'ل'
ALIF = 'ا'

def get_enhanced_weight(char, prev_char=None, next_char=None, position_in_word=0, word_length=1):
    """
    Calculate enhanced phonetic duration weight using makhraj/sifat classification.
    
    Args:
        char: Current character
        prev_char: Previous character (for context)
        next_char: Next character (for context)
        position_in_word: Position of char in word (0-indexed)
        word_length: Total length of word
    
    Returns:
        weight (float): Duration weight for this character
    """
    
    # =================== DIACRITICS ===================
    if char in DIACRITICS:
        if char == SHADDA:
            # Gemination - doubles the letter's inherent duration
            return 2.0
        elif char == SUKUN:
            # Sukun is very short (absence of vowel)
            return 0.2
        elif char in (FATHA, DAMMA, KASRA):
            # Short vowels are brief
            return 0.35
        elif char == ALIF_KHANJARIYYA:
            # Superscript alif - indicates long vowel
            return 0.8
        else:
            # Other diacritical marks
            return 0.3
    
    # =================== MADD LETTERS ===================
    if char in MADD_LETTERS or char == ALIF_WASLA:
        # Alif with Wasla doesn't elongate - it's silent/hamza carrier
        if char == ALIF_WASLA:
            return 0.8
        
        # Check for madd conditions
        if prev_char:
            if char == 'ا' and prev_char == FATHA:
                # Madd tabii'i (natural madd) = 2 harakaat
                return 3.0
            elif char == 'و' and prev_char == DAMMA:
                return 3.0
            elif char == 'ي' and prev_char == KASRA:
                return 3.0
            elif prev_char == SHADDA:
                # After shadda - madd lazim
                return 3.5
        
        # Default madd weight (contextually elongated)
        return 2.5
    
    # =================== THROAT LETTERS (حلق) ===================
    if char in HALQ_LETTERS:
        # Deep articulation from throat takes more time
        if char == 'ء':
            # Hamza - glottal stop, brief
            return 0.9
        elif char == 'ه':
            # Ha - breathy, medium
            return 1.2
        elif char in 'عح':
            # Ayn and Haa - deep throat, longer
            return 1.4
        elif char in 'غخ':
            # Ghayn and Khaa - fricative, longer
            return 1.5
        return 1.3
    
    # =================== EMPHATIC LETTERS (مفخمة) ===================
    if char in MUFAKHKHAM_LETTERS:
        # Heavy letters require more effort to articulate
        if char == 'ق':
            # Qaf - deep uvular, naturally longer
            return 1.4
        elif char in 'صضطظ':
            # Emphatic consonants - heavy articulation
            return 1.3
        return 1.3
    
    # =================== QALQALAH LETTERS ===================
    if char in QALQALAH_LETTERS:
        # Bouncing/echoing sound when stopping on them
        if next_char == SUKUN or position_in_word == word_length - 1:
            # Qalqalah active at word end or with sukun
            return 1.3
        return 1.1
    
    # =================== LIP LETTERS (شفوية) ===================
    if char in SHAFAWI_LETTERS:
        if char == 'ب':
            # Ba - plosive, quick
            return 0.9
        elif char == 'م':
            # Meem - nasal resonance, medium
            return 1.1
        elif char == 'و':
            # Already handled in madd, but as consonant
            return 1.0
        return 1.0
    
    # =================== LIGHT LETTERS (مرققة) ===================
    if char in MURAKKAK_LETTERS:
        if char in 'سش':
            # Sibilants - fricative, slightly longer
            return 1.1
        elif char == 'ل':
            # Lam - variable, default medium
            return 1.0
        elif char == 'ن':
            # Nun - nasal, medium
            return 1.1
        elif char in 'ثذ':
            # Interdentals - distinct, medium
            return 1.0
        return 0.95
    
    # =================== RA (ر) - SPECIAL CASE ===================
    if char == 'ر':
        # Ra has variable weight based on context (tafkhim/tarqiq)
        # For now, treat as medium-heavy
        return 1.2
    
    # =================== DEFAULT ===================
    return 1.0


def distribute_within_word_enhanced(word_text, word_start, word_end):
    """
    Distribute timing across characters using enhanced makhraj/sifat weights.
    """
    if not word_text:
        return []
    
    word_length = len(word_text)
    
    # Calculate weights for each character with context
    weights = []
    for i, char in enumerate(word_text):
        prev_char = word_text[i-1] if i > 0 else None
        next_char = word_text[i+1] if i < len(word_text) - 1 else None
        
        weight = get_enhanced_weight(
            char, 
            prev_char=prev_char,
            next_char=next_char,
            position_in_word=i,
            word_length=word_length
        )
        weights.append((char, weight))
    
    # Normalize weights
    total_weight = sum(w for _, w in weights)
    if total_weight == 0:
        total_weight = 1
    
    word_duration = word_end - word_start
    
    # Generate timings
    timings = []
    current_time = word_start
    
    for i, (char, weight) in enumerate(weights):
        char_duration = (weight / total_weight) * word_duration
        timings.append({
            "char": char,
            "start": current_time,
            "end": current_time + char_duration,
            "weight": weight,
            "category": categorize_char(char)
        })
        current_time += char_duration
    
    return timings


def categorize_char(char):
    """Categorize a character for debugging."""
    if char in DIACRITICS:
        if char == SHADDA:
            return "shadda"
        elif char == SUKUN:
            return "sukun"
        return "haraka"
    elif char in MADD_LETTERS:
        return "madd"
    elif char in HALQ_LETTERS:
        return "halq"
    elif char in MUFAKHKHAM_LETTERS:
        return "mufakhkham"
    elif char in QALQALAH_LETTERS:
        return "qalqalah"
    elif char in SHAFAWI_LETTERS:
        return "shafawi"
    elif char in MURAKKAK_LETTERS:
        return "murakkak"
    elif char == 'ر':
        return "ra"
    return "base"


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    DEVICE = "cpu"
    
    print("=== Enhanced Phonetic Alignment Engine v2 ===")
    print("Using Makhraj/Sifat classification from classical Arabic linguistics")
    print()
    
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
    
    # 3. Distribute within each word using enhanced phonetic weights
    print("Applying enhanced makhraj/sifat weights...")
    all_char_timings = []
    
    for wt in word_timestamps:
        word_text = wt['text']
        word_start = wt['start']
        word_end = wt['end']
        
        # Skip empty words
        if not word_text.strip():
            continue
        
        # Distribute timing within this word
        char_timings = distribute_within_word_enhanced(word_text, word_start, word_end)
        
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
    
    # Print first 20 for verification with categories
    print("\n=== First 20 characters with enhanced weights ===")
    print(f"{'Idx':>4} | {'Char':^4} | {'Duration':>8} | {'Start':>7} | Category")
    print("-" * 50)
    
    for wt in word_timestamps[:3]:  # First 3 words
        word_text = wt['text']
        chars = distribute_within_word_enhanced(word_text, wt['start'], wt['end'])
        for ct in chars:
            dur_ms = (ct['end'] - ct['start']) * 1000
            print(f"{' ':>4} | {ct['char']:^4} | {dur_ms:>6.0f}ms | {ct['start']:>6.3f}s | {ct['category']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
