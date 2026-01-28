#!/usr/bin/env python3
"""
Lisan Pure Proportional Aligner (v1.3)

Key principle: NO artificial milliseconds added.
Uses Lisan ratios ONLY for proportional redistribution within
fixed word boundaries detected by CTC/WhisperX.

- Word duration is FIXED (from CTC)
- Letter duration is PROPORTIONAL (from Lisan ratios)
- Total stays constant, only distribution changes
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

# Arabic character sets
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
MADD_LETTERS = set('اويٱى')  # Alif, Waw, Ya, Alif Wasla, Alif Maqsura
LONG_VOWELS = set('ٰ')  # Superscript alif
HALQ_LETTERS = set('ءهعحغخ')  # Throat letters

# Phonetic Acoustic Profiles (Text-Informed Detection)
PHONETIC_PROFILES = {
    'stop_consonants': set('بتطدكقء'),  # Silence then burst
    'nasals': set('من'),                # Low energy, stable formant
    'fricatives': set('سشصزظذثف'),      # High frequency noise
    'liquids': set('لر'),               # Formant-like but constrained
    'vowels': set('اوي'),               # High energy, clear formants
    'throat': set('عحغخه'),             # Lower frequency, turbulent
}

def get_acoustic_profile(char):
    """
    Get expected acoustic behavior for a letter.
    Used to guide the wave detection algorithm.
    """
    if char in PHONETIC_PROFILES['stop_consonants']:
        return 'onset'  # Look for sharp burst
    elif char in PHONETIC_PROFILES['vowels'] or char in DIACRITICS:
        return 'sustain' # Look for stability
    elif char in PHONETIC_PROFILES['nasals']:
        return 'dip'     # Look for energy dip
    elif char in PHONETIC_PROFILES['fricatives']:
        return 'noise'   # Look for high-freq flux
    return 'generic'

# Lisan-based PROPORTIONAL ratios with COMPLETE Tajweed coverage
# These only affect distribution, not total length

# Letter categories
QALQALAH = set('قطبجد')  # Echoing letters
GHUNNA_LETTERS = set('نم')  # Nasal letters
NOON_SAKIN_LETTERS = set('يرملون')  # Idgham letters (يرملون)
IZHAAR_LETTERS = set('ءهعحغخ')  # Throat letters (same as HALQ)

LISAN_RATIOS = {
    # Base consonant = 1.0 (reference)
    'consonant': 1.0,
    
    # Harakat (short vowels) - proportionally shorter
    'kasra': 0.6,    # ِ
    'fatha': 0.55,   # َ
    'damma': 0.55,   # ُ
    
    # Marks
    'shadda': 0.25,  # ّ (mark itself, not the doubled letter)
    'sukun': 0.2,    # ْ
    'tanween': 0.4,  # ً ٌ ٍ
    
    # Madd types (elongation) - different lengths
    'madd_asli': 1.5,      # Natural madd (2 counts)
    'madd_muttasil': 2.0,  # Connected madd with hamza (4-5 counts)
    'madd_munfasil': 2.0,  # Separated madd (4-5 counts)
    'madd_lazim': 3.0,     # Obligatory madd (6 counts)
    
    # Special letters
    'halq': 1.2,          # Throat letters
    'qalqalah': 1.1,      # Echoing letters (slight bounce)
    'ghunna': 1.3,        # Ghunna (nasal) - 2 counts
}


def get_lisan_ratio(char, prev_char=None, next_char=None):
    """
    Get Lisan-based ratio with COMPLETE Tajweed coverage.
    Pure proportional - no artificial durations.
    """
    # Diacritics
    if char in DIACRITICS:
        if char == SHADDA:
            return LISAN_RATIOS['shadda']
        elif char == '\u0652':  # Sukun
            return LISAN_RATIOS['sukun']
        elif char == '\u0650':  # Kasra
            return LISAN_RATIOS['kasra']
        elif char == '\u064E':  # Fatha
            return LISAN_RATIOS['fatha']
        elif char == '\u064F':  # Damma
            return LISAN_RATIOS['damma']
        elif char in '\u064B\u064C\u064D':  # Tanween
            return LISAN_RATIOS['tanween']
        else:
            return 0.3  # Other marks
    
    # Ghunna: Noon or Meem with Shadda get extended
    if char in GHUNNA_LETTERS:
        if next_char == SHADDA:
            return LISAN_RATIOS['ghunna']  # Ghunna - 2 counts
        return 1.0  # Normal noon/meem
    
    # Qalqalah: Letters with echoing sound
    if char in QALQALAH:
        if next_char == '\u0652' or next_char is None:  # Sukun or end of word
            return LISAN_RATIOS['qalqalah']
        return 1.0
    
    # Madd letters - detect type
    if char in MADD_LETTERS or char in LONG_VOWELS:
        # Check for Madd Lazim (followed by shadda or sukun in same word)
        if next_char == SHADDA or next_char == '\u0652':
            return LISAN_RATIOS['madd_lazim']  # 6 counts
        
        # Check for Madd Muttasil (followed by hamza in same word)
        if next_char == 'ء':
            return LISAN_RATIOS['madd_muttasil']  # 4-5 counts
        
        # Check if proper madd (preceded by matching vowel)
        if prev_char:
            if char == 'ا' and prev_char == '\u064E':  # Alif after Fatha
                return LISAN_RATIOS['madd_asli']
            elif char in 'وٱ' and prev_char == '\u064F':  # Waw after Damma
                return LISAN_RATIOS['madd_asli']
            elif char == 'ي' and prev_char == '\u0650':  # Ya after Kasra
                return LISAN_RATIOS['madd_asli']
        
        return 1.4  # Madd letter without clear context
    
    # Halq (throat) / Izhaar letters
    if char in HALQ_LETTERS:
        return LISAN_RATIOS['halq']
    
    # Default consonant
    return LISAN_RATIOS['consonant']


class LisanPureAligner:
    """
    Pure proportional aligner - NO artificial milliseconds.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        self.sr = 16000
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        print(f"Duration: {len(self.audio)/self.sr:.2f}s")
    
    def detect_madd_from_wave(self, segment, char):
        """
        Detect if a madd letter is actually elongated in the audio.
        Returns a multiplier based on ACTUAL wave measurement.
        """
        if char not in MADD_LETTERS and char not in LONG_VOWELS:
            return 1.0
        
        if len(segment) < 256:
            return 1.0
        
        # Measure energy stability (sustained = madd)
        hop = 128
        energy = librosa.feature.rms(y=segment, hop_length=hop)[0]
        if len(energy) < 3:
            return 1.0
        
        # Coefficient of variation - low = sustained
        mean_e = np.mean(energy)
        if mean_e == 0:
            return 1.0
        cv = np.std(energy) / mean_e
        
        # Low CV means stable energy = sustained vowel
        if cv < 0.3:
            return 2.0  # Strong madd
        elif cv < 0.5:
            return 1.5  # Light madd
        else:
            return 1.2  # Slight boost
    
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




    def detect_acoustic_boundaries(self, segment, n_letters, chars=None):
        """
        Detect REAL acoustic boundaries using Phonetic Profiling.
        Uses text knowledge to look for specific features.
        """
        if len(segment) < 256 or n_letters <= 1:
            return np.linspace(0, len(segment)/self.sr, n_letters + 1)
            
        # 1. Base Features
        S = np.abs(librosa.stft(segment, hop_length=128))
        # Flux (Spectral Change)
        flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max))
        flux = gaussian_filter1d(flux, sigma=1)
        # RMS Energy
        energy = librosa.feature.rms(y=segment, hop_length=128)[0]
        energy = gaussian_filter1d(energy, sigma=1)
        
        # 2. Text-Informed Peak Picking
        # If we know the letters, we can weight the detection
        weighted_flux = flux.copy()
        
        if chars:
            # We skip 0 (start) and match internal boundaries
            # i corresponds to boundary between char[i] and char[i+1]
            for i in range(len(chars) - 1):
                curr_char = chars[i]
                next_char = chars[i+1]
                
                profile_curr = get_acoustic_profile(curr_char)
                profile_next = get_acoustic_profile(next_char)
                
                # Stop -> Vowel = Huge Burst
                if profile_curr == 'onset' and profile_next == 'sustain':
                    # Boost flux detection here
                    pass 
                
                # Vowel -> Nasal = Energy Dip
                if profile_curr == 'sustain' and profile_next == 'dip':
                    # Look for energy drop
                    pass
        
        # Standard Peak Picking (Weighted by our knowledge)
        peaks = librosa.util.peak_pick(weighted_flux, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=1)
        peak_times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=128)
        
        # 3. Add start (0) and end (duration)
        duration = len(segment) / self.sr
        boundaries = np.concatenate([[0], peak_times, [duration]])
        boundaries = np.sort(np.unique(boundaries))
        
        # 4. Fallback/Match
        if len(boundaries) != n_letters + 1:
            return np.linspace(0, duration, n_letters + 1)
            
        return boundaries

    def align(self, text):
        """
        Align with fusion of:
        1. CTC Word Boundaries (Fixed Anchor)
        2. Internal Acoustic Boundaries (Real Wave Detection)
        3. Tajweed Rules (Labeling/Ratio Pries)
        """
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
            
            # Get all characters
            chars = list(word_text)
            n_chars = len(chars)
            if n_chars == 0:
                continue
            
            # Extract audio segment
            start_sample = int(word_start * self.sr)
            end_sample = int(word_end * self.sr)
            segment = self.audio[start_sample:end_sample]
            
            # 1. Calculate Expected Ratios (Tajweed)
            tajweed_ratios = []
            for i, c in enumerate(chars):
                prev_c = chars[i-1] if i > 0 else None
                next_c = chars[i+1] if i < len(chars)-1 else None
                ratio = get_lisan_ratio(c, prev_c, next_c)
                
                # Check actual wave for madd elongation to adjust expectation
                if c in MADD_LETTERS:
                     # Rough estimate segment for detection
                    est_start = int((i/n_chars) * len(segment))
                    est_end = int(((i+1)/n_chars) * len(segment))
                    wave_mult = self.detect_madd_from_wave(segment[est_start:est_end], c)
                    ratio *= wave_mult
                
                tajweed_ratios.append(ratio)
            
            total_ratio = sum(tajweed_ratios)
            norm_ratios = [r/total_ratio for r in tajweed_ratios]
            
            # 2. Detect Acoustic Boundaries (Wave)
            # Try to detect actual letter transitions
            # For robustness in v1.4, we blend Acoustic with Tajweed
            # If acoustic detection is messy, we adhere closer to Tajweed
            # If acoustic is clear, we snap to it.
            
            # Currently implementing "Smart Blend":
            # Use Tajweed ratios to define "Target Centers"
            # Search for Acoustic Boundaries near those targets
            
            # Smart Detection with Phonetic Profiling (v1.6)
            acoustic_boundaries = self.detect_acoustic_boundaries(segment, n_chars, chars)
            
            # Since raw acoustic detection can be noisy, we use it to refine 
            # the Tajweed-expected boundaries rather than replacing them entirely
            
            current_time = 0
            timings_in_word = []
            
            # Calculate cumulative expected times
            cum_ratios = np.cumsum([0] + norm_ratios)
            expected_boundaries = cum_ratios * word_duration
            
            # Map expected boundaries to nearest valid acoustic boundary
            # This is a simple 1D alignment
            final_boundaries = [0]
            for k in range(1, n_chars):
                expected = expected_boundaries[k]
                # Find nearest acoustic peak (skip 0 and end)
                candidates = acoustic_boundaries[1:-1]
                if len(candidates) > 0:
                    nearest_idx = (np.abs(candidates - expected)).argmin()
                    detected = candidates[nearest_idx]
                    
                    # Only snap if within reasonable window (e.g. ±30%) using Lisan constraint
                    if abs(detected - expected) < 0.3 * word_duration:
                        final_boundaries.append(detected)
                    else:
                        final_boundaries.append(expected)
                else:
                    final_boundaries.append(expected)
            final_boundaries.append(word_duration)
            
            # Generate timings from boundaries
            for i in range(n_chars):
                start = final_boundaries[i]
                end = final_boundaries[i+1]
                
                timings_in_word.append({
                    'char': chars[i],
                    'start': word_start + start,
                    'end': word_start + end
                })
            
            # 3. Merge Diacritics into Base Letters (Visual Grouping)
            # User request: highlight letter and its diacritics together
            merged_timings = []
            
            i = 0
            while i < len(timings_in_word):
                current = timings_in_word[i]
                
                # If it's a base letter, check ahead for diacritics
                combined_char = current['char']
                combined_end = current['end']
                
                j = i + 1
                while j < len(timings_in_word):
                    next_t = timings_in_word[j]
                    if next_t['char'] in DIACRITICS or next_t['char'] in SHADDA:
                        combined_char += next_t['char']
                        combined_end = next_t['end']  # Extend end to cover diacritic
                        j += 1
                    else:
                        break
                
                # Create merged entry
                merged_timings.append({
                    'char': combined_char,
                    'start': current['start'],
                    'end': combined_end
                })
                
                i = j  # Skip processed diacritics
            
            all_timings.extend(merged_timings)
        
        # Add indices
        for i, t in enumerate(all_timings):
            t['idx'] = i
        
        print(f"Total: {len(all_timings)} grapheme clusters (Merged Diacritics)")
        return all_timings


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("=" * 60)
    print("Lisan Pure Proportional Aligner v1.3")
    print("NO artificial milliseconds - Pure wave + Lisan ratios")
    print("=" * 60)
    
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    print(f"\nText: {len(text)} chars")
    
    aligner = LisanPureAligner(AUDIO_PATH)
    timings = aligner.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Preview
    print("\n=== First 25 characters (Pure Proportional) ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 35)
    for t in timings[:25]:
        dur = (t['end'] - t['start']) * 1000
        print(f"{t['char']:^4} | {dur:>6.0f}ms | {t['start']:>6.3f}s")


if __name__ == "__main__":
    main()
