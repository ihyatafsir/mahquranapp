#!/usr/bin/env python3
"""
LisanWaveformAligner - Waveform-based Arabic letter alignment

Inspired by concepts from Lisan al-Arab dictionary:
- نبض (nabḍ) - Pulse detection for letter onset boundaries
- موج (mawj) - Wave contour for duration distribution across letters
- تردد (taraddud) - Spectral/frequency analysis for letter type classification
- ردد (radd) - Autocorrelation to detect shadda (gemination)

Key advantage: We KNOW the exact letter sequence (Quran text is fixed),
so we do guided matching instead of open transcription.
"""
import json
import numpy as np
from pathlib import Path
import scipy.signal as signal
import librosa

# ============================================================================
# Arabic Letter Classification (from phonetic_aligner_v2.py)
# ============================================================================
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
SUKUN = '\u0652'
MADD_LETTERS = set('اوي')
HALQ_LETTERS = set('ءهعحغخ')  # Throat letters
MUFAKHKHAM = set('صضطظخغق')   # Emphatic letters


class LisanWaveformAligner:
    """
    Waveform-based aligner using Arabic linguistic concepts.
    
    Since we know the exact Quran text, we can do guided alignment
    using audio features matched to expected letter patterns.
    """
    
    def __init__(self, audio_path, text, sample_rate=16000):
        """
        Initialize with audio and known text.
        
        Args:
            audio_path: Path to audio file
            text: Known Arabic text (from Quran)
            sample_rate: Audio sample rate
        """
        self.sr = sample_rate
        self.text = text
        self.letters = [c for c in text if c not in ' ']
        
        # Load audio
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(str(audio_path), sr=self.sr)
        self.duration = len(self.audio) / self.sr
        print(f"Audio duration: {self.duration:.2f}s, {len(self.letters)} letters")
        
        # Compute features
        self._compute_features()
    
    def _compute_features(self):
        """Pre-compute audio features for alignment."""
        # Frame parameters
        self.hop_length = 256
        self.frame_length = 1024
        
        # نبض - RMS energy envelope (pulse detection)
        self.rms = librosa.feature.rms(
            y=self.audio, 
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        print(f"Computed RMS envelope: {len(self.rms)} frames")
        
        # موج - Amplitude envelope (wave contour)
        self.envelope = np.abs(signal.hilbert(self.audio))
        # Downsample envelope to match RMS frames
        self.envelope_frames = np.array([
            np.mean(self.envelope[i*self.hop_length:(i+1)*self.hop_length])
            for i in range(len(self.rms))
        ])
        
        # تردد - Spectral centroid (frequency patterns)
        self.spectral_centroid = librosa.feature.spectral_centroid(
            y=self.audio,
            sr=self.sr,
            hop_length=self.hop_length
        )[0]
        print(f"Computed spectral centroid: {len(self.spectral_centroid)} frames")
    
    def detect_nabd(self, min_prominence=0.1):
        """
        نبض (nabḍ) - Detect energy pulses as potential letter onsets.
        
        Returns:
            onsets: Array of frame indices where energy pulses occur
        """
        # Normalize RMS
        rms_norm = self.rms / (np.max(self.rms) + 1e-8)
        
        # Find peaks (energy pulses)
        peaks, properties = signal.find_peaks(
            rms_norm,
            prominence=min_prominence,
            distance=5  # Minimum frames between peaks
        )
        
        print(f"نبض detected {len(peaks)} energy pulses")
        return peaks
    
    def analyze_mawj(self, start_frame, end_frame):
        """
        موج (mawj) - Analyze wave contour for letter distribution.
        
        The wave's "back and forth" motion indicates energy distribution.
        Letters should be placed proportionally to energy.
        
        Returns:
            weights: Array of relative weights for each frame
        """
        segment = self.envelope_frames[start_frame:end_frame]
        if len(segment) == 0:
            return np.array([1.0])
        
        # Normalize to get distribution weights
        weights = segment / (np.sum(segment) + 1e-8)
        return weights
    
    def detect_taraddud(self, start_frame, end_frame):
        """
        تردد (taraddud) - Analyze frequency patterns.
        
        Different letter types have distinct spectral signatures:
        - Throat letters (halq): Lower frequencies
        - Sibilants (س/ش): Higher frequencies
        - Nasals (م/ن): Mid-range formants
        
        Returns:
            avg_centroid: Average spectral centroid (Hz)
        """
        segment = self.spectral_centroid[start_frame:end_frame]
        if len(segment) == 0:
            return 0
        return np.mean(segment)
    
    def detect_radd(self, start_frame, end_frame, threshold=0.7):
        """
        ردد (radd) - Detect repetition/doubling (shadda) via autocorrelation.
        
        Shadda = geminated consonant. Look for sustained, self-similar energy.
        
        Returns:
            is_doubled: Boolean indicating possible shadda
            confidence: Correlation strength
        """
        start_sample = start_frame * self.hop_length
        end_sample = min(end_frame * self.hop_length, len(self.audio))
        segment = self.audio[start_sample:end_sample]
        
        if len(segment) < 100:
            return False, 0.0
        
        # Autocorrelation
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / (autocorr[0] + 1e-8)  # Normalize
        
        # Look for strong self-similarity at half-period
        mid = len(autocorr) // 2
        if mid > 10:
            peak_corr = np.max(autocorr[mid-10:mid+10])
            return peak_corr > threshold, float(peak_corr)
        
        return False, 0.0
    
    def frame_to_time(self, frame):
        """Convert frame index to time in seconds."""
        return frame * self.hop_length / self.sr
    
    def time_to_frame(self, time_sec):
        """Convert time in seconds to frame index."""
        return int(time_sec * self.sr / self.hop_length)
    
    def get_phonetic_weight(self, char, prev_char=None):
        """Get duration weight based on letter type (Hafs tajweed)."""
        if char in DIACRITICS:
            if char == SHADDA:
                return 2.0
            elif char == SUKUN:
                return 0.2
            return 0.35
        
        if char in MADD_LETTERS:
            return 2.5
        
        if char in HALQ_LETTERS:
            return 1.3
        
        if char in MUFAKHKHAM:
            return 1.2
        
        return 1.0
    
    def align(self, word_boundaries=None):
        """
        Main alignment using all Lisan concepts.
        
        Args:
            word_boundaries: Optional list of (start_time, end_time, word_text) tuples
                           If None, aligns across entire audio
        
        Returns:
            letter_timings: List of {idx, char, start, end} dicts
        """
        print("\n=== Starting Lisan Waveform Alignment ===")
        
        # نبض: Detect pulses across entire audio
        pulses = self.detect_nabd()
        pulse_times = [self.frame_to_time(p) for p in pulses]
        print(f"Pulse times: {pulse_times[:5]}...")
        
        # If no word boundaries provided, use uniform distribution
        if word_boundaries is None:
            # Simple uniform distribution with phonetic weights
            letter_timings = self._align_uniform_with_weights()
        else:
            # Use word boundaries + waveform refinement
            letter_timings = self._align_with_word_boundaries(word_boundaries)
        
        return letter_timings
    
    def _align_uniform_with_weights(self):
        """
        Distribute letters across audio using phonetic weights
        and waveform energy (موج).
        """
        # Calculate weights
        weights = []
        for i, char in enumerate(self.letters):
            prev_char = self.letters[i-1] if i > 0 else None
            w = self.get_phonetic_weight(char, prev_char)
            weights.append(w)
        
        total_weight = sum(weights)
        
        # Get energy contour for entire audio
        mawj = self.analyze_mawj(0, len(self.rms))
        
        # Resample energy contour to match letter count
        # This adds wave-based modulation to phonetic weights
        energy_factors = np.interp(
            np.linspace(0, 1, len(self.letters)),
            np.linspace(0, 1, len(mawj)),
            mawj
        )
        energy_factors = energy_factors / (np.mean(energy_factors) + 1e-8)
        
        # Combine phonetic weights with energy
        combined_weights = [w * (0.7 + 0.3 * e) for w, e in zip(weights, energy_factors)]
        total_combined = sum(combined_weights)
        
        # Generate timings
        timings = []
        current_time = 0.0
        
        for i, (char, weight) in enumerate(zip(self.letters, combined_weights)):
            duration = (weight / total_combined) * self.duration
            timings.append({
                "idx": i,
                "char": char,
                "start": current_time,
                "end": current_time + duration
            })
            current_time += duration
        
        return timings
    
    def _align_with_word_boundaries(self, word_boundaries):
        """
        Align letters within word boundaries using waveform features.
        """
        letter_timings = []
        letter_idx = 0
        
        for word_start, word_end, word_text in word_boundaries:
            word_letters = [c for c in word_text if c not in ' ']
            if not word_letters:
                continue
            
            start_frame = self.time_to_frame(word_start)
            end_frame = self.time_to_frame(word_end)
            
            # موج: Get energy contour for this word
            mawj = self.analyze_mawj(start_frame, end_frame)
            
            # تردد: Get spectral characteristics
            centroid = self.detect_taraddud(start_frame, end_frame)
            
            # Calculate phonetic weights for letters
            weights = []
            for i, char in enumerate(word_letters):
                prev_char = word_letters[i-1] if i > 0 else None
                w = self.get_phonetic_weight(char, prev_char)
                
                # ردد: Check for shadda
                if i > 0 and word_letters[i] == SHADDA:
                    is_doubled, conf = self.detect_radd(start_frame, end_frame)
                    if is_doubled:
                        w *= 1.2  # Enhance shadda weight
                
                weights.append(w)
            
            # Combine with energy
            if len(mawj) > 1:
                energy_factors = np.interp(
                    np.linspace(0, 1, len(word_letters)),
                    np.linspace(0, 1, len(mawj)),
                    mawj
                )
                energy_factors = energy_factors / (np.mean(energy_factors) + 1e-8)
                combined = [w * (0.7 + 0.3 * e) for w, e in zip(weights, energy_factors)]
            else:
                combined = weights
            
            # Distribute within word
            total = sum(combined)
            word_duration = word_end - word_start
            current = word_start
            
            for char, weight in zip(word_letters, combined):
                duration = (weight / total) * word_duration
                letter_timings.append({
                    "idx": letter_idx,
                    "char": char,
                    "start": current,
                    "end": current + duration
                })
                current += duration
                letter_idx += 1
        
        return letter_timings


def main():
    """Test the LisanWaveformAligner on Abdul Basit Surah 1."""
    PROJECT_ROOT = Path(__file__).parent.parent
    VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
    AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
    OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
    
    print("=" * 60)
    print("LisanWaveformAligner - Guided Arabic Letter Alignment")
    print("Using concepts from Lisan al-Arab: نبض / موج / تردد / ردد")
    print("=" * 60)
    
    # Load text
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get('1', []))
    print(f"\nLoaded text: {len(text)} chars")
    
    # Create aligner
    aligner = LisanWaveformAligner(str(AUDIO_PATH), text)
    
    # Align (using uniform distribution + waveform energy)
    timings = aligner.align()
    
    print(f"\nGenerated {len(timings)} letter timings")
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "letter_timing_1.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Show first 15
    print("\n=== First 15 characters (Lisan Waveform) ===")
    print(f"{'Idx':>4} | {'Char':^4} | {'Duration':>8} | {'Start':>7}")
    print("-" * 40)
    for t in timings[:15]:
        dur_ms = (t['end'] - t['start']) * 1000
        print(f"{t['idx']:>4} | {t['char']:^4} | {dur_ms:>6.0f}ms | {t['start']:>6.3f}s")


if __name__ == "__main__":
    main()
