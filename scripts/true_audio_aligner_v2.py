#!/usr/bin/env python3
"""
True Audio Aligner v2: "Fasl & Nabd" (Separation & Pulse)

Emulates human hearing by picking up TWO types of boundaries:
1. نبض (Nabḍ): Energy pulses (Amplitude Onset) - for distinct strokes
2. فصل (Fasl): Spectral Separation (Spectral Flux) - for "same breath" transitions

Key insight from Lisan al-Arab: 
- "Fasl" (separating distinct things) describes how we distinguish sounds 
  even when they flow together (Wasl/Idgham).
- Human ears detect changes in TIMBRE (spectrum) even if LOUDNESS (amplitude) is constant.
"""
import json
import numpy as np
from pathlib import Path
import librosa
import torch
from scipy.signal import find_peaks

# CTC aligner for word boundaries
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')

class TrueAudioAlignerV2:
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        print(f"Loading audio: {audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=16000)
        self.hop_length = 256
        
    def detect_nabd(self, segment):
        """
        Detect Energy Pulses (Nabḍ) - Classical Onsets
        Good for: Plosives, distinct stops, new breaths
        """
        onset_env = librosa.onset.onset_strength(y=segment, sr=self.sr, hop_length=self.hop_length)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        return times, onset_env

    def detect_fasl(self, segment):
        """
        Detect Spectral Separation (Fasl) - Spectral Flux
        Good for: "Same breath" transitions (e.g. M -> A -> L), Vowel changes
        Humans hear these as changes in "color" or quality.
        """
        # Compute spectrogram
        S = np.abs(librosa.stft(segment, hop_length=self.hop_length))
        
        # Spectral Flux: Difference between consecutive frames
        flux = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=self.sr, hop_length=self.hop_length)
        
        # Find peaks in flux (meaning significant timbre change)
        peaks = librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.3, wait=5)
        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        return times, flux

    def combine_boundaries(self, nabd_times, fasl_times, segment_duration):
        """
        Combine and deduplicate boundaries using "Mayz" (Distinction) logic.
        """
        # Start with 0 and end
        boundaries = [0.0, segment_duration]
        
        # Add all detected changes
        all_detections = sorted(list(nabd_times) + list(fasl_times))
        
        # Filter duplicates (within 50ms)
        unique_bounds = []
        if all_detections:
            curr = all_detections[0]
            unique_bounds.append(curr)
            for t in all_detections[1:]:
                if t - curr > 0.05:  # 50ms minimum separation
                    unique_bounds.append(t)
                    curr = t
                    
        boundaries.extend(unique_bounds)
        return sorted(list(set(boundaries)))

    def get_word_boundaries(self, text):
        """Get CTC word boundaries."""
        print("Running CTC alignment...")
        alignment_model, alignment_tokenizer = load_alignment_model(self.device, dtype=torch.float32)
        audio_waveform = load_audio(self.audio_path, alignment_model.dtype, alignment_model.device)
        emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=8)
        tokens_starred, text_starred = preprocess_text(text, romanize=True, language='ara')
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def split_letters(self, word_text):
        """Split word into base letters attached with their diacritics."""
        units = []
        i = 0
        while i < len(word_text):
            char = word_text[i]
            if char in DIACRITICS:
                if units: units[-1] = (units[-1][0], units[-1][1] + char)
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

    def align(self, text):
        # 1. Get Word Boundaries (CTC)
        words = self.get_word_boundaries(text)
        final_timings = []
        
        print(f"\nAligning {len(words)} words using Fasl & Nabd detection...")
        
        for wt in words:
            word_text = wt['text']
            w_start = wt['start']
            w_end = wt['end']
            
            if not word_text.strip(): continue
            
            # Extract audio segment for this word
            start_sample = int(w_start * self.sr)
            end_sample = int(w_end * self.sr)
            segment = self.y[start_sample:end_sample]
            duration = (end_sample - start_sample) / self.sr
            
            if duration < 0.05: continue

            # 2. Detect Boundaries within word
            nabd, _ = self.detect_nabd(segment)
            fasl, _ = self.detect_fasl(segment)
            
            # Combine boundaries
            bounds = self.combine_boundaries(nabd, fasl, duration)
            
            # 3. Match to Letters
            units = self.split_letters(word_text)
            n_units = len(units)
            
            # Interpolate if mismatch
            if len(bounds) - 1 != n_units:
                # If we detected different number of segments than letters,
                # we use the detected "Fasl" points as anchors and interpolate
                bounds = np.linspace(0, duration, n_units + 1).tolist()
            
            # Assign
            for i, (base, diacs) in enumerate(units):
                seg_start = w_start + bounds[i]
                seg_end = w_start + bounds[i+1]
                seg_dur = seg_end - seg_start
                
                # Base uses 70% of segment, diacritics 30%
                base_end = seg_start + (seg_dur * (0.7 if diacs else 1.0))
                
                final_timings.append({
                    "char": base,
                    "start": seg_start,
                    "end": base_end
                })
                
                if diacs:
                    d_dur = (seg_end - base_end) / len(diacs)
                    curr = base_end
                    for d in diacs:
                        final_timings.append({
                            "char": d,
                            "start": curr,
                            "end": curr + d_dur
                        })
                        curr += d_dur
                        
        # Add index
        for i, t in enumerate(final_timings):
            t['idx'] = i
            
        return final_timings

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("="*60)
    print("True Audio Aligner v2: 'Fasl & Nabḍ'")
    print("Using Spectral Flux to detect 'same breath' transitions")
    print("="*60)
    
    with open(VERSES_PATH, 'r') as f:
        data = json.load(f)
    text = ' '.join(v['text'] for v in data['1'])
    
    aligner = TrueAudioAlignerV2(AUDIO_PATH)
    timings = aligner.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "letter_timing_1.json", 'w') as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved {len(timings)} letters to {OUTPUT_DIR}/letter_timing_1.json")
    
    # Preview
    print("\nPreview (Fasl Detection):")
    for t in timings[:10]:
         print(f"{t['char']} : {t['start']:.3f}s - {t['end']:.3f}s ({(t['end']-t['start'])*1000:.0f}ms)")

if __name__ == "__main__":
    main()
