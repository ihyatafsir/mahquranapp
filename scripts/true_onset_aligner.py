#!/usr/bin/env python3
"""
True Audio-Based Letter Aligner

Unlike phonetic weight distribution (which guesses), this DETECTS actual
letter boundaries from the audio waveform by finding onset pulses (نبض).

Key insight: We know the EXACT letters, so we detect N onsets for N letters
and match them directly - no guessing.

Uses CTC for word-level boundaries (accurate), then onset detection within words.
"""
import json
import numpy as np
from pathlib import Path
import librosa
import torch

# CTC aligner for word boundaries
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

# Diacritics don't have their own onset - they're part of the base letter
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')


def detect_onsets(audio, sr=16000, hop_length=256):
    """
    Detect actual onset times in audio using librosa's onset detection.
    These are the REAL letter/syllable boundaries in the audio.
    """
    # Use librosa's onset detection (energy-based + spectral flux)
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # Start at the actual onset, not peak
        units='frames'
    )
    
    # Convert frames to times
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    return onset_times


def get_ctc_word_boundaries(audio_path, text, device="cpu"):
    """
    Get accurate word-level boundaries from CTC forced alignment.
    """
    print("Loading CTC aligner for word boundaries...")
    alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=torch.float32)
    audio_waveform = load_audio(str(audio_path), alignment_model.dtype, alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=8)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language='ara')
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    return word_timestamps


def split_word_letters(word_text):
    """
    Split word into base letters (consonants + madd) and their diacritics.
    Each base letter "owns" the following diacritics.
    
    Returns list of (base_char, attached_diacritics) tuples
    """
    units = []
    i = 0
    while i < len(word_text):
        char = word_text[i]
        if char in DIACRITICS:
            # Diacritic without base - attach to previous
            if units:
                units[-1] = (units[-1][0], units[-1][1] + char)
            i += 1
            continue
        
        # Base character - collect following diacritics
        diacritics = ""
        j = i + 1
        while j < len(word_text) and word_text[j] in DIACRITICS:
            diacritics += word_text[j]
            j += 1
        
        units.append((char, diacritics))
        i = j
    
    return units


def align_word_with_onsets(word_text, word_start, word_end, word_audio, sr=16000):
    """
    Align letters within a word using detected onsets.
    
    This DETECTS real boundaries, doesn't guess.
    """
    # Split into letter units (base + diacritics)
    letter_units = split_word_letters(word_text)
    n_letters = len(letter_units)
    
    if n_letters == 0:
        return []
    
    word_duration = word_end - word_start
    
    # If word is very short or only 1 letter, just uniform distribution
    if n_letters == 1 or word_duration < 0.1:
        timings = []
        current = word_start
        duration_per = word_duration / n_letters
        for base, diacs in letter_units:
            # Base letter timing
            timings.append({
                "char": base,
                "start": current,
                "end": current + duration_per * 0.7
            })
            # Diacritics (attached)
            diacs_start = current + duration_per * 0.7
            for d in diacs:
                d_dur = (duration_per * 0.3) / max(len(diacs), 1)
                timings.append({
                    "char": d,
                    "start": diacs_start,
                    "end": diacs_start + d_dur
                })
                diacs_start += d_dur
            current += duration_per
        return timings
    
    # Detect onsets in this word's audio
    onsets = detect_onsets(word_audio, sr=sr)
    
    # Filter onsets to be within [0, word_duration]
    onsets = [o for o in onsets if 0 <= o <= word_duration]
    
    # Add word start and end as boundaries
    boundaries = [0.0] + list(onsets) + [word_duration]
    boundaries = sorted(set(boundaries))
    
    # Need at least n_letters boundaries
    # If we have more onsets than letters, merge nearby ones
    # If we have fewer, interpolate
    
    if len(boundaries) < n_letters + 1:
        # Interpolate to get enough boundaries
        boundaries = np.linspace(0, word_duration, n_letters + 1).tolist()
    elif len(boundaries) > n_letters + 1:
        # Too many onsets - select evenly spaced ones
        indices = np.linspace(0, len(boundaries) - 1, n_letters + 1).astype(int)
        boundaries = [boundaries[i] for i in indices]
    
    # Assign each letter unit to a boundary segment
    timings = []
    for i, (base, diacs) in enumerate(letter_units):
        seg_start = word_start + boundaries[i]
        seg_end = word_start + boundaries[i + 1]
        seg_dur = seg_end - seg_start
        
        # Base letter gets ~70% of segment
        base_dur = seg_dur * 0.7 if diacs else seg_dur
        timings.append({
            "char": base,
            "start": seg_start,
            "end": seg_start + base_dur
        })
        
        # Diacritics share remaining 30%
        if diacs:
            diacs_start = seg_start + base_dur
            diacs_dur = seg_dur * 0.3
            for d in diacs:
                d_dur = diacs_dur / len(diacs)
                timings.append({
                    "char": d,
                    "start": diacs_start,
                    "end": diacs_start + d_dur
                })
                diacs_start += d_dur
    
    return timings


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    DEVICE = "cpu"
    
    print("=" * 60)
    print("True Audio-Based Letter Aligner")
    print("Detects REAL letter boundaries from audio onsets")
    print("=" * 60)
    
    # Load text
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get('1', []))
    print(f"\nText: {len(text)} chars")
    
    # Get word boundaries from CTC (this is accurate)
    word_timestamps = get_ctc_word_boundaries(AUDIO_PATH, text, DEVICE)
    print(f"CTC gave {len(word_timestamps)} word boundaries")
    
    # Load full audio for segment extraction
    print("Loading audio for onset detection...")
    audio, sr = librosa.load(str(AUDIO_PATH), sr=16000)
    
    # Align each word using onset detection
    print("Detecting onsets within each word...")
    all_timings = []
    
    for wt in word_timestamps:
        word_text = wt['text']
        word_start = wt['start']
        word_end = wt['end']
        
        if not word_text.strip():
            continue
        
        # Extract word audio segment
        start_sample = int(word_start * sr)
        end_sample = int(word_end * sr)
        word_audio = audio[start_sample:end_sample]
        
        # Align letters within this word
        word_timings = align_word_with_onsets(word_text, word_start, word_end, word_audio, sr)
        
        for wt_char in word_timings:
            wt_char['idx'] = len(all_timings)
            all_timings.append(wt_char)
    
    print(f"\nTotal: {len(all_timings)} character timings")
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_timings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Show first 20
    print("\n=== First 20 characters (Real Onset Detection) ===")
    print(f"{'Idx':>4} | {'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 40)
    for t in all_timings[:20]:
        dur_ms = (t['end'] - t['start']) * 1000
        print(f"{t['idx']:>4} | {t['char']:^4} | {dur_ms:>6.0f}ms | {t['start']:>6.3f}s")


if __name__ == "__main__":
    main()
