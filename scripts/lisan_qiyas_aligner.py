#!/usr/bin/env python3
"""
Lisan Qiyas (قياس) Hybrid Aligner

Hybrid approach inspired by Lisan al-Arab concepts:
- قياس (qiyas): Proportional measurement - redistribute within fixed bounds
- وزن (wazn): Balance/weight - letter type ratios
- مقدار (miqdar): Quantity - actual wave measurement

Key principle: 
- Use REAL wave boundaries (no artificial time addition)
- Apply قياس (proportion) within each segment
- The total duration is FIXED by wave detection
- We redistribute that FIXED duration using letter ratios

This preserves the actual audio timing while ensuring 
each letter gets its proportional share.
"""
import json
import numpy as np
from pathlib import Path
import librosa
from scipy.ndimage import gaussian_filter1d
import torch

from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
MADD_LETTERS = set('اوي')
HALQ_LETTERS = set('ءهعحغخ')

# Lisan-based RATIOS (not absolute durations)
# These are relative weights for proportional distribution
LISAN_RATIOS = {
    'base_consonant': 1.0,    # Base unit
    'haraka': 0.5,            # Short vowels are ~half a consonant
    'sukun': 0.2,             # Very brief mark
    'shadda_mark': 0.3,       # The mark itself
    'madd_letter': 2.0,       # Elongation letters get 2x
    'halq': 1.3,              # Throat letters slightly longer
}


def get_ratio(char, prev_char=None):
    """Get Lisan ratio for a character."""
    if char in DIACRITICS:
        if char == SHADDA:
            return LISAN_RATIOS['shadda_mark']
        elif char == '\u0652':  # Sukun
            return LISAN_RATIOS['sukun']
        else:
            return LISAN_RATIOS['haraka']
    
    if char in MADD_LETTERS:
        return LISAN_RATIOS['madd_letter']
    
    if char in HALQ_LETTERS:
        return LISAN_RATIOS['halq']
    
    return LISAN_RATIOS['base_consonant']


class LisanQiyasAligner:
    """
    Hybrid aligner using wave detection + Lisan proportions.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        self.sr = 16000
        self.hop_length = 256
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        print(f"Duration: {len(self.audio)/self.sr:.2f}s")
    
    def measure_wave_energy(self, segment):
        """
        Measure the energy contour of a segment.
        Returns energy values that can be used as weights.
        """
        energy = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
        return energy
    
    def detect_wave_boundaries(self, segment, n_units):
        """
        Detect n_units boundaries from waveform.
        Uses onset detection + energy peaks.
        """
        if len(segment) < 512 or n_units <= 0:
            return np.linspace(0, len(segment)/self.sr, n_units + 1)
        
        duration = len(segment) / self.sr
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr, hop_length=self.hop_length)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length,
            backtrack=True, units='time'
        )
        
        # Combine with 0 and duration
        boundaries = np.unique(np.concatenate([[0], onsets, [duration]]))
        boundaries.sort()
        
        # Select exactly n_units + 1 boundaries
        if len(boundaries) == n_units + 1:
            return boundaries
        elif len(boundaries) < n_units + 1:
            # Interpolate
            return np.linspace(0, duration, n_units + 1)
        else:
            # Select evenly
            indices = np.round(np.linspace(0, len(boundaries) - 1, n_units + 1)).astype(int)
            return boundaries[indices]
    
    def split_word(self, word_text):
        """Split word into base letter units with diacritics."""
        units = []
        i = 0
        while i < len(word_text):
            char = word_text[i]
            if char in DIACRITICS:
                if units:
                    units[-1]['diacritics'] += char
                i += 1
                continue
            
            unit = {'base': char, 'diacritics': ''}
            j = i + 1
            while j < len(word_text) and word_text[j] in DIACRITICS:
                unit['diacritics'] += word_text[j]
                j += 1
            units.append(unit)
            i = j
        return units
    
    def apply_qiyas(self, all_chars, total_duration, start_time):
        """
        Apply قياس (proportional distribution) to characters.
        Total duration is FIXED - we just redistribute it.
        """
        # Calculate ratios
        ratios = []
        for i, c in enumerate(all_chars):
            prev = all_chars[i-1] if i > 0 else None
            ratios.append(get_ratio(c, prev))
        
        # Apply shadda doubling to preceding consonant's effective ratio
        for i, c in enumerate(all_chars):
            if c == SHADDA and i > 0:
                # The preceding letter gets doubled effect
                ratios[i-1] *= 2.0
        
        # Normalize to sum to 1
        total_ratio = sum(ratios)
        norm_ratios = [r / total_ratio for r in ratios]
        
        # Distribute duration proportionally
        timings = []
        current = start_time
        for c, ratio in zip(all_chars, norm_ratios):
            char_dur = total_duration * ratio
            timings.append({
                'char': c,
                'start': current,
                'end': current + char_dur
            })
            current += char_dur
        
        return timings
    
    def get_word_boundaries(self, text):
        """Get CTC word boundaries."""
        print("Getting word boundaries from CTC...")
        model, tokenizer = load_alignment_model(self.device, dtype=torch.float32)
        waveform = load_audio(self.audio_path, model.dtype, model.device)
        emissions, stride = generate_emissions(model, waveform, batch_size=8)
        tokens, text_starred = preprocess_text(text, romanize=True, language='ara')
        segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
        spans = get_spans(tokens, segments, blank)
        return postprocess_results(text_starred, spans, stride, scores)
    
    def align(self, text):
        """Full hybrid alignment."""
        words = self.get_word_boundaries(text)
        print(f"Got {len(words)} words")
        
        all_timings = []
        
        for wt in words:
            word_text = wt['text']
            word_start = wt['start']
            word_end = wt['end']
            word_duration = word_end - word_start
            
            if not word_text.strip() or word_duration <= 0:
                continue
            
            # Get all characters in word
            all_chars = list(word_text)
            if not all_chars:
                continue
            
            # Apply قياس (proportional distribution) - NO added time
            char_timings = self.apply_qiyas(all_chars, word_duration, word_start)
            all_timings.extend(char_timings)
        
        # Add indices
        for i, t in enumerate(all_timings):
            t['idx'] = i
        
        print(f"Total: {len(all_timings)} characters")
        return all_timings


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("=" * 60)
    print("Lisan Qiyas (قياس) Hybrid Aligner")
    print("Wave boundaries + Proportional distribution")
    print("=" * 60)
    
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    print(f"\nText: {len(text)} chars")
    
    aligner = LisanQiyasAligner(AUDIO_PATH)
    timings = aligner.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Preview
    print("\n=== First 20 characters (Qiyas Hybrid) ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 30)
    for t in timings[:20]:
        dur = (t['end'] - t['start']) * 1000
        print(f"{t['char']:^4} | {dur:>6.0f}ms | {t['start']:>6.3f}s")


if __name__ == "__main__":
    main()
