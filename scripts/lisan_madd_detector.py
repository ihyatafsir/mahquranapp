#!/usr/bin/env python3
"""
Lisan Madd Detector - The Final Puzzle

The missing piece: detecting ACTUAL madd (مد) duration from waveform.

Lisan concepts:
- يمدّ ويقصر (extends and shortens) - madd letters vary in length
- استمرّ (sustains) - sustaining a vowel sound

Key insight: Madd letters (ا/و/ي) sustain a constant energy/pitch.
We can detect these SUSTAINED regions in the waveform by looking for:
1. Low spectral change (same sound continuing)
2. Stable energy (not decaying quickly like consonants)
3. Stable pitch/fundamental frequency

This measures the ACTUAL reciter's elongation, not estimated weights.
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
MADD_LETTERS = set('اويٱ')  # Including alif wasla
LONG_VOWELS = set('ٰ')  # Superscript alif (already indicates long)
HALQ_LETTERS = set('ءهعحغخ')


class LisanMaddDetector:
    """
    Detects actual madd (elongation) duration from waveform.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        self.sr = 16000
        self.hop_length = 256
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        print(f"Duration: {len(self.audio)/self.sr:.2f}s")
    
    def detect_sustained_regions(self, segment):
        """
        Detect regions where sound is SUSTAINED (استمرّ).
        These are likely madd letters being held by the reciter.
        
        Returns: array of sustain scores per frame (higher = more sustained)
        """
        if len(segment) < 512:
            return np.zeros(1)
        
        # 1. Compute spectral flux (low flux = sustained sound)
        S = np.abs(librosa.stft(segment, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)
        
        # Invert: high score where flux is LOW (sustained sound)
        max_flux = np.max(flux) if np.max(flux) > 0 else 1
        sustain_score = 1 - (flux / max_flux)
        
        # 2. Check energy stability (sustained sounds have stable RMS)
        energy = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
        energy = gaussian_filter1d(energy.astype(np.float64), sigma=2)
        
        # Energy stability: low variance in local windows
        stability = np.zeros_like(energy)
        window = 5
        for i in range(window, len(energy) - window):
            local_std = np.std(energy[max(0, i-window):i+window])
            local_mean = np.mean(energy[max(0, i-window):i+window])
            if local_mean > 0:
                stability[i] = 1 - min(local_std / local_mean, 1)
        
        # Pad stability to match sustain_score length
        min_len = min(len(sustain_score), len(stability))
        sustain_score = sustain_score[:min_len]
        stability = stability[:min_len]
        
        # Combined score: both low flux AND stable energy = sustained vowel
        combined = sustain_score * stability
        
        return combined
    
    def measure_madd_duration(self, segment, letter, position_in_word):
        """
        Measure actual madd duration for a letter based on sustain analysis.
        Returns a multiplier relative to base duration.
        """
        if letter not in MADD_LETTERS and letter not in LONG_VOWELS:
            return 1.0  # Not a madd letter
        
        sustain_scores = self.detect_sustained_regions(segment)
        
        if len(sustain_scores) == 0:
            return 1.0
        
        # Average sustain in this segment
        avg_sustain = np.mean(sustain_scores)
        
        # High sustain (> 0.6) means reciter is holding this vowel
        if avg_sustain > 0.6:
            return 2.5  # Strong madd
        elif avg_sustain > 0.4:
            return 2.0  # Medium madd
        elif avg_sustain > 0.2:
            return 1.5  # Light madd
        else:
            return 1.0  # No significant madd detected
    
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
    
    def split_word(self, word_text):
        """Split word into letter units."""
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
    
    def get_base_ratio(self, char):
        """Get base ratio for character type."""
        if char in DIACRITICS:
            if char == SHADDA:
                return 0.4
            elif char == '\u0652':  # Sukun
                return 0.3
            else:  # Harakat
                return 0.6
        
        if char in HALQ_LETTERS:
            return 1.3
        
        return 1.0
    
    def align(self, text):
        """Full alignment with madd detection."""
        words = self.get_word_boundaries(text)
        print(f"Got {len(words)} words")
        print("Detecting madd (sustained vowel) regions...")
        
        all_timings = []
        
        for wt in words:
            word_text = wt['text']
            word_start = wt['start']
            word_end = wt['end']
            word_duration = word_end - word_start
            
            if not word_text.strip() or word_duration <= 0:
                continue
            
            # Extract audio segment
            start_sample = int(word_start * self.sr)
            end_sample = int(word_end * self.sr)
            segment = self.audio[start_sample:end_sample]
            
            # Get all characters
            all_chars = list(word_text)
            n_chars = len(all_chars)
            
            if n_chars == 0:
                continue
            
            # First pass: calculate base ratios
            ratios = []
            for i, c in enumerate(all_chars):
                ratio = self.get_base_ratio(c)
                
                # Detect madd: check if this is a madd letter
                if c in MADD_LETTERS or c in LONG_VOWELS:
                    # Estimate position in segment
                    approx_start = int((i / n_chars) * len(segment))
                    approx_end = int(((i + 1) / n_chars) * len(segment))
                    char_segment = segment[approx_start:approx_end]
                    
                    # Measure actual madd duration
                    madd_mult = self.measure_madd_duration(char_segment, c, i)
                    ratio *= madd_mult
                
                # Shadda doubles the preceding consonant
                if c == SHADDA and i > 0 and all_chars[i-1] not in DIACRITICS:
                    ratios[i-1] *= 2.0
                
                ratios.append(ratio)
            
            # Normalize and distribute
            total_ratio = sum(ratios)
            if total_ratio == 0:
                total_ratio = 1
            
            current = word_start
            for c, ratio in zip(all_chars, ratios):
                char_dur = (ratio / total_ratio) * word_duration
                all_timings.append({
                    'char': c,
                    'start': current,
                    'end': current + char_dur
                })
                current += char_dur
        
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
    print("Lisan Madd Detector - The Final Puzzle")
    print("Detecting ACTUAL sustained vowel duration")
    print("=" * 60)
    
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    print(f"\nText: {len(text)} chars")
    
    detector = LisanMaddDetector(AUDIO_PATH)
    timings = detector.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Preview - show madd letters specifically
    print("\n=== First 25 characters (Madd Detection) ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7} | Madd?")
    print("-" * 45)
    for t in timings[:25]:
        dur = (t['end'] - t['start']) * 1000
        is_madd = "✓" if t['char'] in MADD_LETTERS or dur > 100 else ""
        print(f"{t['char']:^4} | {dur:>6.0f}ms | {t['start']:>6.3f}s | {is_madd}")


if __name__ == "__main__":
    main()
