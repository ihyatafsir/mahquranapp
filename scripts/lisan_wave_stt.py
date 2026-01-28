#!/usr/bin/env python3
"""
سامع الكاتب (Sami' al-Katib) - The Listening Scribe

A custom Arabic Wave STT system for Quranic letter-level alignment.
Based on phonetic principles from Lisan al-Arab dictionary.

Architecture:
1. سامع (Sami') - Listener: Detects acoustic events from waveform
2. كاتب (Katib) - Scribe: Maps events to known letters using DTW
3. مقرئ (Muqri') - Reciter: Applies tajweed timing constraints

Key insight: We KNOW the exact letter sequence (Quran text is fixed),
so we detect events then ALIGN them, rather than recognizing letters.
"""
import json
import numpy as np
from pathlib import Path
import librosa
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import torch

# CTC for word boundaries
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)

# ============================================================================
# ARABIC PHONETIC CONSTANTS (from Lisan al-Arab)
# ============================================================================
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
SUKUN = '\u0652'
FATHA, DAMMA, KASRA = '\u064E', '\u064F', '\u0650'
MADD_LETTERS = set('اوي')
HALQ_LETTERS = set('ءهعحغخ')
MUFAKHKHAM = set('صضطظخغق')

# Duration weights based on Lisan phonetic principles
LISAN_WEIGHTS = {
    'haraka': 1.0,      # Short vowel (fatha, damma, kasra) - Boosted for visibility
    'sukun': 0.5,       # Silence mark - Boosted to ensure it's seen
    'shadda': 0.6,      # Mark itself - needs to be visible
    'madd': 2.5,        # Elongation - Multiplier of base
    'halq': 1.4,        # Throat letters - Slightly longer
    'mufakhkham': 1.3,  # Emphatic - Slightly longer
    'normal': 1.0,      # Default consonant
}


class Sami:
    """سامع (Listener) - Multi-feature acoustic event detector."""
    
    def __init__(self, sr=16000, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_onsets(self, audio):
        """Detect energy pulses (نبض)."""
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        peaks = librosa.onset.onset_detect(
            y=audio, sr=self.sr, hop_length=self.hop_length,
            backtrack=True, units='time'
        )
        return peaks, onset_env
    
    def detect_mfcc_transitions(self, audio):
        """Detect formant transitions (مخرج changes)."""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=13, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Sum absolute delta across coefficients
        transition_strength = np.sum(np.abs(mfcc_delta), axis=0)
        transition_strength = gaussian_filter1d(transition_strength, sigma=2)
        
        # Find peaks
        peaks = librosa.util.peak_pick(
            transition_strength,
            pre_max=3, post_max=3, pre_avg=3, post_avg=5,
            delta=np.mean(transition_strength) * 0.5, wait=5
        )
        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        return times, transition_strength
    
    def detect_spectral_flux(self, audio):
        """Detect timbre changes (فصل)."""
        S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
        
        # Spectral flux: frame-to-frame difference
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])  # Pad to match length
        flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)  # Ensure float64
        
        # Find peaks using onset_detect style (more robust)
        threshold = np.mean(flux) + 0.5 * np.std(flux)
        peaks = np.where(flux > threshold)[0]
        
        # Filter to local maxima
        local_max = []
        for i in range(1, len(peaks) - 1):
            p = peaks[i]
            if flux[p] >= flux[p-1] and flux[p] >= flux[p+1]:
                local_max.append(p)
        
        times = librosa.frames_to_time(np.array(local_max), sr=self.sr, hop_length=self.hop_length)
        return times, flux
    
    def fuse_detections(self, onset_times, mfcc_times, flux_times, min_gap=0.03):
        """
        Fuse multiple detection methods into unified event timestamps.
        Uses weighted voting and minimum gap filtering.
        """
        # Combine all detections
        all_times = []
        
        # Weight: onsets are most reliable, then MFCC, then flux
        for t in onset_times:
            all_times.append((t, 1.0))  # High confidence
        for t in mfcc_times:
            all_times.append((t, 0.7))  # Medium confidence
        for t in flux_times:
            all_times.append((t, 0.5))  # Lower confidence
        
        if not all_times:
            return np.array([])
        
        # Sort by time
        all_times.sort(key=lambda x: x[0])
        
        # Merge nearby detections (within min_gap)
        merged = []
        i = 0
        while i < len(all_times):
            cluster_times = [all_times[i][0]]
            cluster_weights = [all_times[i][1]]
            j = i + 1
            
            while j < len(all_times) and all_times[j][0] - cluster_times[0] < min_gap:
                cluster_times.append(all_times[j][0])
                cluster_weights.append(all_times[j][1])
                j += 1
            
            # Weighted average of cluster
            total_weight = sum(cluster_weights)
            weighted_time = sum(t * w for t, w in zip(cluster_times, cluster_weights)) / total_weight
            merged.append((weighted_time, total_weight))
            i = j
        
        # Return times only
        return np.array([t for t, w in merged])
    
    def detect_all(self, audio):
        """Run all detection methods and fuse results."""
        onset_times, _ = self.detect_onsets(audio)
        mfcc_times, _ = self.detect_mfcc_transitions(audio)
        flux_times, _ = self.detect_spectral_flux(audio)
        
        fused = self.fuse_detections(onset_times, mfcc_times, flux_times)
        return fused


class Katib:
    """كاتب (Scribe) - Maps acoustic events to known letters."""
    
    def __init__(self):
        pass
    
    def get_letter_units(self, word_text):
        """
        Split word into letter units (base letter + attached diacritics).
        Each unit corresponds to one acoustic event.
        """
        units = []
        i = 0
        while i < len(word_text):
            char = word_text[i]
            
            if char in DIACRITICS:
                # Diacritic without base - attach to previous
                if units:
                    units[-1]['diacritics'] += char
                i += 1
                continue
            
            # Base character - collect following diacritics
            unit = {'base': char, 'diacritics': ''}
            j = i + 1
            while j < len(word_text) and word_text[j] in DIACRITICS:
                unit['diacritics'] += word_text[j]
                j += 1
            
            units.append(unit)
            i = j
        
        return units
    
    def compute_alignment_cost(self, n_letters, n_events):
        """
        Compute cost matrix for aligning letters to events.
        Uses simple positional cost (prefer sequential alignment).
        """
        if n_events == 0 or n_letters == 0:
            return np.ones((n_letters, n_events + 1))
        
        # Create positional cost: prefer letters[i] → events[j] where i/n ≈ j/m
        cost = np.zeros((n_letters, n_events + 1))
        for i in range(n_letters):
            for j in range(n_events + 1):
                # Ideal position for letter i
                ideal_j = (i / n_letters) * n_events
                cost[i, j] = abs(j - ideal_j) / max(n_events, 1)
        
        return cost
    
    def dtw_align(self, n_letters, events, word_start, word_end):
        """
        Use DTW-style greedy alignment to assign letter boundaries.
        Since events may not equal letters, we interpolate.
        """
        duration = word_end - word_start
        
        # Filter events to be within word boundaries
        word_events = [e - word_start for e in events if word_start <= e <= word_end]
        
        # Add start and end as boundaries
        boundaries = [0.0] + word_events + [duration]
        boundaries = sorted(set(boundaries))
        
        # If we have exactly right number of boundaries
        if len(boundaries) == n_letters + 1:
            return boundaries
        
        # Need to adjust: interpolate to get exactly n_letters + 1 boundaries
        if len(boundaries) < n_letters + 1:
            # Not enough events - interpolate between existing ones
            boundaries = np.linspace(0, duration, n_letters + 1).tolist()
        else:
            # Too many events - select best subset
            # Use dynamic programming or simple selection
            indices = np.round(np.linspace(0, len(boundaries) - 1, n_letters + 1)).astype(int)
            boundaries = [boundaries[i] for i in indices]
        
        return boundaries
    
    def align_word(self, word_text, word_start, word_end, events):
        """Align letters in a word to detected events."""
        units = self.get_letter_units(word_text)
        n_letters = len(units)
        
        if n_letters == 0:
            return []
        
        # Get boundaries
        boundaries = self.dtw_align(n_letters, events, word_start, word_end)
        
        # Create timings for each unit
        timings = []
        for i, unit in enumerate(units):
            seg_start = word_start + boundaries[i]
            seg_end = word_start + boundaries[i + 1]
            
            timings.append({
                'base': unit['base'],
                'diacritics': unit['diacritics'],
                'start': seg_start,
                'end': seg_end
            })
        
        return timings


class Muqri:
    """مقرئ (Reciter) - Applies tajweed timing constraints."""
    
    def __init__(self, haraka_unit=0.05):
        """
        haraka_unit: Base duration for one haraka (beat unit) in seconds.
        Typical range: 0.04-0.08s depending on recitation speed.
        """
        self.haraka_unit = haraka_unit
    
    def get_weight(self, char, prev_char=None):
        """Get duration weight based on Lisan phonetic rules."""
        if char in DIACRITICS:
            if char == SHADDA:
                return LISAN_WEIGHTS['shadda']
            elif char == SUKUN:
                return LISAN_WEIGHTS['sukun']
            elif char in (FATHA, DAMMA, KASRA):
                return LISAN_WEIGHTS['haraka']
            else:
                return 0.3
        
        if char in MADD_LETTERS:
            # Check for madd condition
            if prev_char and prev_char in (FATHA, DAMMA, KASRA):
                return LISAN_WEIGHTS['madd']
            return 2.0
        
        if char in HALQ_LETTERS:
            return LISAN_WEIGHTS['halq']
        
        if char in MUFAKHKHAM:
            return LISAN_WEIGHTS['mufakhkham']
        
        return LISAN_WEIGHTS['normal']
    
    def expand_to_characters(self, word_timings):
        """
        Expand unit timings to individual character timings.
        Distributes time between base letter and diacritics based on weights.
        """
        char_timings = []
        
        for unit in word_timings:
            base = unit['base']
            diacs = unit['diacritics']
            start = unit['start']
            end = unit['end']
            duration = end - start
            
            # Calculate weights
            all_chars = [base] + list(diacs)
            weights = []
            for i, c in enumerate(all_chars):
                prev = all_chars[i-1] if i > 0 else None
                weights.append(self.get_weight(c, prev))
            
            # Apply shadda doubling to base
            if SHADDA in diacs:
                weights[0] *= 2.0
            
            total_weight = sum(weights)
            
            # Distribute duration
            current = start
            for c, w in zip(all_chars, weights):
                char_dur = (w / total_weight) * duration
                char_timings.append({
                    'char': c,
                    'start': current,
                    'end': current + char_dur
                })
                current += char_dur
        
        return char_timings
    
    def apply_constraints(self, char_timings):
        """Apply minimum duration constraints."""
        # Minimum duration to ensure visibility on 60fps screens (~16ms per frame)
        # 45ms ensures at least ~2-3 frames of visibility
        VISUAL_MIN_DURATION = 0.045
        
        for ct in char_timings:
            char = ct['char']
            
            # Theoretical minimum based on phonetic weight
            theory_min = self.haraka_unit * self.get_weight(char)
            
            # Use the larger of visual min or theoretical min
            target_min = max(theory_min, VISUAL_MIN_DURATION)
            
            actual_dur = ct['end'] - ct['start']
            
            # If current duration is less than target, extend it
            if actual_dur < target_min:
                # We extend the end, but need to be careful not to overlap too much
                # Ideally we should re-distribute, but expanding is safer for visibility
                ct['end'] = ct['start'] + target_min
        
        # After expanding, we might have created overlaps. 
        # A proper layout engine would resolve these, but for now simple expansion 
        # ensures visibility even if it slightly eats into next letter's start relative to audio.
        # However, the frontend uses binary search on START time, so actually
        # extends "active" duration until next start.
        # So we actually need to adjust START times if we want strict non-overlap, 
        # OR just rely on the fact that frontend usually does:
        # active = (time >= char.start && time < next_char.start)
        
        # IF frontend uses next_char.start as cutoff, then simply extending .end 
        # doesn't help unless we shift the NEXT letters.
        # Let's adjust boundaries to ensure minimum spacing.
        
        for i in range(len(char_timings) - 1):
            curr = char_timings[i]
            next_char = char_timings[i+1]
            
            dur = next_char['start'] - curr['start']
            if dur < VISUAL_MIN_DURATION:
                # Current letter is too short because next one starts too soon
                # Push next letter forward
                diff = VISUAL_MIN_DURATION - dur
                next_char['start'] += diff
                next_char['end'] += diff
                
                # Propagate shift to subsequent letters to avoid overlap bunching
                for j in range(i+2, len(char_timings)):
                    char_timings[j]['start'] += diff
                    char_timings[j]['end'] += diff

        return char_timings


class LisanWaveSTT:
    """
    Main class: سامع الكاتب (The Listening Scribe)
    Combines Sami', Katib, and Muqri' for full letter alignment.
    """
    
    def __init__(self, audio_path, device="cpu"):
        self.audio_path = str(audio_path)
        self.device = device
        
        # Load audio
        print(f"Loading audio: {audio_path}")
        self.audio, self.sr = librosa.load(self.audio_path, sr=16000)
        self.duration = len(self.audio) / self.sr
        print(f"Duration: {self.duration:.2f}s")
        
        # Initialize components
        self.sami = Sami(sr=self.sr)
        self.katib = Katib()
        self.muqri = Muqri()
    
    def get_word_boundaries(self, text):
        """Get CTC word boundaries (accurate at word level)."""
        print("Getting word boundaries from CTC...")
        model, tokenizer = load_alignment_model(self.device, dtype=torch.float32)
        waveform = load_audio(self.audio_path, model.dtype, model.device)
        emissions, stride = generate_emissions(model, waveform, batch_size=8)
        tokens, text_starred = preprocess_text(text, romanize=True, language='ara')
        segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
        spans = get_spans(tokens, segments, blank)
        return postprocess_results(text_starred, spans, stride, scores)
    
    def align(self, text):
        """Full alignment pipeline."""
        # 1. Get word boundaries
        words = self.get_word_boundaries(text)
        print(f"Got {len(words)} word boundaries")
        
        # 2. Detect acoustic events across entire audio
        print("Detecting acoustic events (سامع)...")
        all_events = self.sami.detect_all(self.audio)
        print(f"Detected {len(all_events)} acoustic events")
        
        # 3. Align each word
        print("Aligning letters to events (كاتب)...")
        all_timings = []
        
        for wt in words:
            word_text = wt['text']
            word_start = wt['start']
            word_end = wt['end']
            
            if not word_text.strip():
                continue
            
            # Get events within this word
            word_events = all_events[(all_events >= word_start) & (all_events <= word_end)]
            
            # Align
            unit_timings = self.katib.align_word(word_text, word_start, word_end, word_events)
            
            # Expand to characters with timing constraints
            char_timings = self.muqri.expand_to_characters(unit_timings)
            char_timings = self.muqri.apply_constraints(char_timings)
            
            all_timings.extend(char_timings)
        
        # Add indices
        for i, ct in enumerate(all_timings):
            ct['idx'] = i
        
        print(f"Total: {len(all_timings)} character timings")
        return all_timings


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("=" * 60)
    print("سامع الكاتب - Lisan Wave STT System")
    print("The Listening Scribe for Quranic Letter Alignment")
    print("=" * 60)
    
    # Load text
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get('1', []))
    print(f"\nText: {len(text)} chars")
    
    # Create aligner and run
    stt = LisanWaveSTT(AUDIO_PATH)
    timings = stt.align(text)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Preview
    print("\n=== First 20 characters ===")
    print(f"{'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 30)
    for ct in timings[:20]:
        dur = (ct['end'] - ct['start']) * 1000
        print(f"{ct['char']:^4} | {dur:>6.0f}ms | {ct['start']:>6.3f}s")


if __name__ == "__main__":
    main()
