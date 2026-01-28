#!/usr/bin/env python3
"""
True Audio Aligner v3: "Makhraj (مخرج) Detection"

Uses FORMANT ANALYSIS to detect letter transitions based on their
unique articulation points (مخارج الحروف).

From Lisan al-Arab:
- "الغانذ: الحَلْق ومخرج الصوت" (The throat and the exit point of sound)
- "الميم مُطْبقَة" (Meem is closed - unique formant pattern)
- "لقرب مخرجيهما" (Due to closeness of their articulation points)

Each Arabic letter has distinct F1/F2 formant patterns:
- Throat letters (ع/ح/خ): Low F1, Very Low F2
- Lip letters (ب/م/و): Low F1, Medium F2  
- Front letters (ي/ك/ش): High F2
- Back letters (ق/غ): Low F2

This analyzes formant trajectories to find letter boundaries.
"""
import json
import numpy as np
from pathlib import Path
import librosa
import torch
from scipy.ndimage import gaussian_filter1d

# CTC for word boundaries
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')


class MakhrajAligner:
    """
    Makhraj-based aligner using formant trajectories.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        print(f"Loading audio: {audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=16000)
        self.hop_length = 256
        self.n_fft = 2048
        
    def compute_mfcc_trajectory(self, segment):
        """
        Compute MFCC trajectory - captures formant-like patterns.
        MFCCs are good proxies for articulation point changes.
        """
        mfcc = librosa.feature.mfcc(
            y=segment, 
            sr=self.sr, 
            n_mfcc=13,
            hop_length=self.hop_length
        )
        return mfcc
    
    def compute_spectral_contrast(self, segment):
        """
        Spectral contrast captures the difference between peaks and valleys
        in the spectrum - good for detecting makhraj changes.
        """
        contrast = librosa.feature.spectral_contrast(
            y=segment,
            sr=self.sr,
            hop_length=self.hop_length
        )
        return contrast
    
    def detect_makhraj_changes(self, segment):
        """
        Detect مخرج (articulation point) changes using combined features:
        1. MFCC delta (formant trajectory changes)
        2. Spectral contrast changes
        3. ZCR changes (voiced/unvoiced)
        """
        if len(segment) < self.n_fft:
            return []
        
        # 1. MFCC delta - captures formant transitions
        mfcc = self.compute_mfcc_trajectory(segment)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_change = np.sum(np.abs(mfcc_delta), axis=0)  # Sum across coefficients
        
        # 2. Spectral contrast - captures resonance changes
        contrast = self.compute_spectral_contrast(segment)
        contrast_delta = np.diff(contrast, axis=1)
        contrast_change = np.sum(np.abs(contrast_delta), axis=0)
        
        # Pad to same length
        min_len = min(len(mfcc_change), len(contrast_change))
        mfcc_change = mfcc_change[:min_len]
        contrast_change = contrast_change[:min_len]
        
        # 3. Combine signals (weighted)
        combined = 0.6 * (mfcc_change / (np.max(mfcc_change) + 1e-8)) + \
                   0.4 * (contrast_change / (np.max(contrast_change) + 1e-8))
        
        # Smooth to avoid noise
        combined = gaussian_filter1d(combined, sigma=2)
        
        # Find peaks (makhraj transition points)
        peaks = librosa.util.peak_pick(
            combined,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
            delta=0.15,  # Lower threshold for more sensitivity
            wait=5
        )
        
        # Convert to times
        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        return times
    
    def detect_energy_onsets(self, segment):
        """
        Standard energy onset detection (نبض).
        """
        onset_env = librosa.onset.onset_strength(
            y=segment, sr=self.sr, hop_length=self.hop_length
        )
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3,
            pre_avg=3, post_avg=5, delta=0.5, wait=10
        )
        return librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
    
    def get_ctc_words(self, text):
        """Get CTC word boundaries."""
        print("Running CTC alignment...")
        model, tokenizer = load_alignment_model(self.device, dtype=torch.float32)
        waveform = load_audio(self.audio_path, model.dtype, model.device)
        emissions, stride = generate_emissions(model, waveform, batch_size=8)
        tokens, text_starred = preprocess_text(text, romanize=True, language='ara')
        segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
        spans = get_spans(tokens, segments, blank)
        return postprocess_results(text_starred, spans, stride, scores)
    
    def split_letters(self, word_text):
        """Split word into letter units (base + diacritics)."""
        units = []
        i = 0
        while i < len(word_text):
            char = word_text[i]
            if char in DIACRITICS:
                if units:
                    units[-1] = (units[-1][0], units[-1][1] + char)
                i += 1
                continue
            diacs = ""
            j = i + 1
            while j < len(word_text) and word_text[j] in DIACRITICS:
                diacs += word_text[j]
                j += 1
            units.append((char, diacs))
            i = j
        return units
    
    def combine_and_match_boundaries(self, makhraj_times, onset_times, n_letters, duration):
        """
        Combine مخرج and نبض boundaries, then match to letter count.
        Uses "Mayz" (distinction) logic to merge nearby boundaries.
        """
        # Combine all detected boundaries
        all_bounds = sorted(set([0.0] + list(makhraj_times) + list(onset_times) + [duration]))
        
        # Filter duplicates (within 30ms - closer tolerance for precision)
        unique = [all_bounds[0]]
        for t in all_bounds[1:]:
            if t - unique[-1] > 0.03:  # 30ms minimum
                unique.append(t)
        
        # If we have more boundaries than needed, keep the strongest
        if len(unique) - 1 > n_letters:
            # Prioritize makhraj-detected boundaries (more precise)
            makhraj_set = set(makhraj_times)
            
            # Score each boundary by closeness to makhraj detection
            scored = []
            for b in unique[1:-1]:  # Exclude 0 and duration
                is_makhraj = any(abs(b - m) < 0.05 for m in makhraj_set)
                scored.append((b, 1.0 if is_makhraj else 0.5))
            
            # Keep top n_letters boundaries
            scored.sort(key=lambda x: -x[1])
            kept = sorted([s[0] for s in scored[:n_letters - 1]])
            unique = [0.0] + kept + [duration]
        
        # If we have fewer, interpolate
        if len(unique) - 1 < n_letters:
            unique = list(np.linspace(0, duration, n_letters + 1))
        
        return unique
    
    def align(self, text):
        """Main alignment using Makhraj detection."""
        words = self.get_ctc_words(text)
        timings = []
        
        print(f"\nAligning {len(words)} words using Makhraj (مخرج) detection...")
        
        for wt in words:
            word_text = wt['text']
            w_start = wt['start']
            w_end = wt['end']
            
            if not word_text.strip():
                continue
            
            # Extract segment
            start_sample = int(w_start * self.sr)
            end_sample = int(w_end * self.sr)
            segment = self.y[start_sample:end_sample]
            duration = (end_sample - start_sample) / self.sr
            
            if duration < 0.05:
                continue
            
            # Split into letters
            units = self.split_letters(word_text)
            n_letters = len(units)
            
            if n_letters == 0:
                continue
            
            # Detect boundaries using both methods
            makhraj_times = self.detect_makhraj_changes(segment)
            onset_times = self.detect_energy_onsets(segment)
            
            # Combine and match
            bounds = self.combine_and_match_boundaries(
                makhraj_times, onset_times, n_letters, duration
            )
            
            # Assign to letters
            for i, (base, diacs) in enumerate(units):
                seg_start = w_start + bounds[i]
                seg_end = w_start + bounds[i + 1]
                seg_dur = seg_end - seg_start
                
                # Base letter gets ~70%, diacritics 30%
                base_ratio = 0.7 if diacs else 1.0
                base_end = seg_start + seg_dur * base_ratio
                
                timings.append({
                    "char": base,
                    "start": seg_start,
                    "end": base_end
                })
                
                if diacs:
                    d_start = base_end
                    d_dur = (seg_end - base_end) / len(diacs)
                    for d in diacs:
                        timings.append({
                            "char": d,
                            "start": d_start,
                            "end": d_start + d_dur
                        })
                        d_start += d_dur
        
        # Add indices
        for i, t in enumerate(timings):
            t['idx'] = i
        
        return timings


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("=" * 60)
    print("True Audio Aligner v3: Makhraj (مخرج) Detection")
    print("Using MFCC + Spectral Contrast for formant-based precision")
    print("=" * 60)
    
    with open(VERSES_PATH, 'r') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    
    aligner = MakhrajAligner(AUDIO_PATH)
    timings = aligner.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "letter_timing_1.json", 'w') as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(timings)} letters to {OUTPUT_DIR}/letter_timing_1.json")
    
    # Preview
    print("\nPreview (Makhraj Detection):")
    for t in timings[:15]:
        dur = (t['end'] - t['start']) * 1000
        print(f"{t['char']:>2} : {t['start']:.3f}s - {t['end']:.3f}s ({dur:.0f}ms)")


if __name__ == "__main__":
    main()
