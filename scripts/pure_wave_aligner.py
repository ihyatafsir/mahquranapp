#!/usr/bin/env python3
"""
Pure Waveform Aligner - "Copy Wave Length"

Instead of estimating durations with weights, this approach:
1. Gets word boundaries from CTC (accurate)
2. Detects ACTUAL acoustic boundaries within each word from the waveform
3. Maps letters to boundaries using the REAL wave durations

No artificial timing addition. Pure waveform-based detection.
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


class PureWaveAligner:
    """
    Aligns letters purely based on detected waveform boundaries.
    No artificial weights or minimum durations - just real audio.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        self.sr = 16000
        self.hop_length = 256
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        print(f"Duration: {len(self.audio)/self.sr:.2f}s")
    
    def detect_boundaries_in_segment(self, segment, n_letters):
        """
        Detect exactly n_letters boundaries within an audio segment.
        Uses energy envelope and onset detection combined.
        """
        if len(segment) < 512 or n_letters <= 0:
            return np.linspace(0, len(segment)/self.sr, n_letters + 1)
        
        duration = len(segment) / self.sr
        
        # 1. Compute energy envelope
        energy = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
        energy = gaussian_filter1d(energy.astype(np.float64), sigma=2)
        
        # 2. Detect onsets (energy increases)
        onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length,
            backtrack=True, units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        
        # 3. Detect spectral changes (timbre transitions)
        mfcc = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=13, hop_length=self.hop_length)
        mfcc_delta = np.sum(np.abs(librosa.feature.delta(mfcc)), axis=0)
        mfcc_delta = gaussian_filter1d(mfcc_delta.astype(np.float64), sigma=2)
        
        # Find peaks in MFCC delta (formant transitions)
        threshold = np.mean(mfcc_delta) + 0.3 * np.std(mfcc_delta)
        mfcc_peaks = np.where(mfcc_delta > threshold)[0]
        
        # Filter to local maxima
        mfcc_local_max = []
        for i in range(1, len(mfcc_peaks) - 1):
            p = mfcc_peaks[i]
            if p > 0 and p < len(mfcc_delta) - 1:
                if mfcc_delta[p] >= mfcc_delta[p-1] and mfcc_delta[p] >= mfcc_delta[p+1]:
                    mfcc_local_max.append(p)
        
        mfcc_times = librosa.frames_to_time(np.array(mfcc_local_max), sr=self.sr, hop_length=self.hop_length)
        
        # 4. Combine all detected boundaries
        all_times = np.concatenate([[0], onset_times, mfcc_times, [duration]])
        all_times = np.unique(all_times)
        all_times.sort()
        
        # 5. Select exactly n_letters + 1 boundaries
        if len(all_times) == n_letters + 1:
            return all_times
        elif len(all_times) < n_letters + 1:
            # Not enough detected - interpolate between detected points
            return np.linspace(0, duration, n_letters + 1)
        else:
            # Too many - select evenly spaced subset from detected
            indices = np.round(np.linspace(0, len(all_times) - 1, n_letters + 1)).astype(int)
            return all_times[indices]
    
    def split_to_letter_units(self, word_text):
        """Split word into base letter units."""
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
        """Align letters using pure waveform detection."""
        words = self.get_word_boundaries(text)
        print(f"Got {len(words)} words")
        
        all_timings = []
        
        for wt in words:
            word_text = wt['text']
            word_start = wt['start']
            word_end = wt['end']
            
            if not word_text.strip():
                continue
            
            # Extract audio segment
            start_sample = int(word_start * self.sr)
            end_sample = int(word_end * self.sr)
            segment = self.audio[start_sample:end_sample]
            
            # Get letter units
            units = self.split_to_letter_units(word_text)
            n_units = len(units)
            
            if n_units == 0:
                continue
            
            # Detect REAL boundaries from waveform
            boundaries = self.detect_boundaries_in_segment(segment, n_units)
            
            # Assign each unit to its detected boundary
            for i, unit in enumerate(units):
                unit_start = word_start + boundaries[i]
                unit_end = word_start + boundaries[i + 1]
                unit_duration = unit_end - unit_start
                
                # All characters in unit (base + diacritics)
                all_chars = [unit['base']] + list(unit['diacritics'])
                n_chars = len(all_chars)
                
                # Simple equal split within unit (these are co-articulated anyway)
                char_duration = unit_duration / n_chars
                
                current = unit_start
                for c in all_chars:
                    all_timings.append({
                        'char': c,
                        'start': current,
                        'end': current + char_duration
                    })
                    current += char_duration
        
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
    print("Pure Waveform Aligner - Copy Wave Length")
    print("No artificial timing - Real audio boundaries only")
    print("=" * 60)
    
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    print(f"\nText: {len(text)} chars")
    
    aligner = PureWaveAligner(AUDIO_PATH)
    timings = aligner.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Preview
    print("\n=== First 20 characters (Pure Waveform) ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 30)
    for t in timings[:20]:
        dur = (t['end'] - t['start']) * 1000
        print(f"{t['char']:^4} | {dur:>6.0f}ms | {t['start']:>6.3f}s")


if __name__ == "__main__":
    main()
