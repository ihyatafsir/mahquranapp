#!/usr/bin/env python3
"""
Batch Lisan Wave STT v2.0

Batch processing version of lisan_wave_stt.py for Abdul Basit surahs.
Uses existing word boundaries from timing data instead of CTC.

Key changes from original:
1. Uses pre-computed word boundaries from timing files
2. Preserves original time units (ms or seconds)
3. Supports batch processing of all surahs
"""
import json
import numpy as np
from pathlib import Path
import argparse

try:
    import librosa
    from scipy.ndimage import gaussian_filter1d
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("Warning: librosa not available")

# Arabic constants
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
SUKUN = '\u0652'
FATHA, DAMMA, KASRA = '\u064E', '\u064F', '\u0650'
MADD_LETTERS = set('اويٱى')
HALQ_LETTERS = set('ءهعحغخ')

# Lisan weights
LISAN_WEIGHTS = {
    'haraka': 1.0, 'sukun': 0.5, 'shadda': 0.6, 'madd': 2.5,
    'halq': 1.4, 'mufakhkham': 1.3, 'normal': 1.0,
}


class Sami:
    """سامع (Listener) - Acoustic event detector."""
    
    def __init__(self, sr=16000, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_onsets(self, audio):
        """Detect energy pulses."""
        if len(audio) < 512:
            return np.array([])
        try:
            peaks = librosa.onset.onset_detect(
                y=audio, sr=self.sr, hop_length=self.hop_length,
                backtrack=True, units='time'
            )
            return peaks
        except Exception:
            return np.array([])
    
    def detect_spectral_flux(self, audio):
        """Detect timbre changes."""
        if len(audio) < 512:
            return np.array([])
        try:
            S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            flux = np.concatenate([[0], flux])
            flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)
            
            threshold = np.mean(flux) + 0.5 * np.std(flux)
            peaks = np.where(flux > threshold)[0]
            
            # Local maxima only
            local_max = []
            for i in range(1, len(peaks) - 1):
                p = peaks[i]
                if flux[p] >= flux[p-1] and flux[p] >= flux[p+1]:
                    local_max.append(p)
            
            times = librosa.frames_to_time(np.array(local_max), sr=self.sr, hop_length=self.hop_length)
            return times
        except Exception:
            return np.array([])
    
    def detect_all(self, audio, word_start_time=0):
        """Detect events and return absolute times."""
        onset_times = self.detect_onsets(audio)
        flux_times = self.detect_spectral_flux(audio)
        
        # Combine (simple union)
        all_times = np.concatenate([onset_times, flux_times])
        all_times = np.unique(all_times)
        all_times.sort()
        
        # Convert to absolute times
        return all_times + word_start_time


class Katib:
    """كاتب (Scribe) - Maps events to letter boundaries."""
    
    def dtw_align(self, n_letters, events, word_duration):
        """Align letters to detected events using interpolation."""
        if n_letters <= 0:
            return [0, word_duration]
        
        # Add boundaries
        boundaries = [0.0] + list(events) + [word_duration]
        boundaries = sorted(set(boundaries))
        
        if len(boundaries) == n_letters + 1:
            return boundaries
        
        if len(boundaries) < n_letters + 1:
            # Interpolate
            return np.linspace(0, word_duration, n_letters + 1).tolist()
        else:
            # Select subset
            indices = np.round(np.linspace(0, len(boundaries) - 1, n_letters + 1)).astype(int)
            return [boundaries[i] for i in indices]


class Muqri:
    """مقرئ (Reciter) - Applies Tajweed weights."""
    
    def get_weight(self, char, prev_char=None):
        if char in DIACRITICS:
            if char == SHADDA: return LISAN_WEIGHTS['shadda']
            elif char == SUKUN: return LISAN_WEIGHTS['sukun']
            elif char in (FATHA, DAMMA, KASRA): return LISAN_WEIGHTS['haraka']
            return 0.3
        if char in MADD_LETTERS: return LISAN_WEIGHTS['madd']
        if char in HALQ_LETTERS: return LISAN_WEIGHTS['halq']
        return LISAN_WEIGHTS['normal']
    
    def redistribute_with_weights(self, letters, boundaries, word_start, unit_scale):
        """Redistribute letter timings using Tajweed weights."""
        n_letters = len(letters)
        if n_letters == 0:
            return letters
        
        # Calculate weights
        chars = [l['char'] for l in letters]
        weights = [self.get_weight(c, chars[i-1] if i > 0 else None) for i, c in enumerate(chars)]
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1
        
        # Use boundaries if matching, else distribute by weight
        if len(boundaries) == n_letters + 1:
            for i, letter in enumerate(letters):
                letter['start'] = (word_start + boundaries[i]) * unit_scale
                letter['end'] = (word_start + boundaries[i+1]) * unit_scale
                letter['duration'] = letter['end'] - letter['start']
        else:
            # Fallback: weight-based distribution
            word_duration = boundaries[-1] - boundaries[0] if len(boundaries) > 1 else 1.0
            current = word_start
            for i, (letter, w) in enumerate(zip(letters, weights)):
                dur = (w / total_weight) * word_duration
                letter['start'] = current * unit_scale
                letter['end'] = (current + dur) * unit_scale
                letter['duration'] = dur * unit_scale
                current += dur
        
        return letters


def detect_units(data):
    """Detect if data is in ms or seconds."""
    if not data:
        return 1
    first_val = data[0].get('start', 0) or data[0].get('end', 0)
    return 1000 if first_val > 100 else 1


def inject_word_idx(data, gap_threshold):
    """Inject wordIdx based on timing gaps."""
    if not data:
        return data
    word_idx = 0
    data[0]['wordIdx'] = word_idx
    for i in range(1, len(data)):
        gap = data[i].get('start', 0) - data[i-1].get('end', 0)
        if gap > gap_threshold:
            word_idx += 1
        data[i]['wordIdx'] = word_idx
    return data


def merge_diacritics(letters):
    """Merge diacritics with base letters."""
    merged = []
    i = 0
    while i < len(letters):
        current = letters[i].copy()
        combined = current['char']
        end = current['end']
        j = i + 1
        while j < len(letters) and letters[j]['char'] in DIACRITICS:
            combined += letters[j]['char']
            end = letters[j]['end']
            j += 1
        current['char'] = combined
        current['end'] = end
        current['duration'] = end - current['start']
        merged.append(current)
        i = j
    return merged


def process_surah(input_path, output_path, audio_path, merge_diacr=True):
    """Process a single surah with wave detection."""
    # Load timing data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print(f"  Skipping {input_path} - empty")
        return False
    
    # Detect units
    unit_scale = detect_units(data)
    unit_name = "ms" if unit_scale == 1000 else "sec"
    
    # Inject wordIdx if missing
    has_word_idx = 'wordIdx' in data[0]
    if not has_word_idx:
        gap_threshold = 50 if unit_scale == 1000 else 0.05
        data = inject_word_idx(data, gap_threshold)
        print(f"    (injected wordIdx)")
    
    # Load audio
    audio = None
    sr = 16000
    if audio_path and Path(audio_path).exists() and WAVE_AVAILABLE:
        try:
            audio, sr = librosa.load(str(audio_path), sr=sr)
        except Exception as e:
            print(f"    Warning: Could not load audio: {e}")
    
    # Group by word
    words = {}
    for letter in data:
        idx = letter['wordIdx']
        if idx not in words:
            words[idx] = []
        words[idx].append(letter)
    
    # Process
    sami = Sami(sr=sr)
    katib = Katib()
    muqri = Muqri()
    
    all_letters = []
    wave_count = 0
    
    for word_idx in sorted(words.keys()):
        word_letters = words[word_idx]
        if not word_letters:
            continue
        
        # Get word boundaries in seconds
        word_start_sec = word_letters[0]['start'] / unit_scale
        word_end_sec = word_letters[-1]['end'] / unit_scale
        word_duration_sec = word_end_sec - word_start_sec
        
        if audio is not None and word_duration_sec > 0.01:
            # Extract audio segment
            start_sample = int(word_start_sec * sr)
            end_sample = int(word_end_sec * sr)
            
            if 0 <= start_sample < end_sample <= len(audio):
                word_audio = audio[start_sample:end_sample]
                
                # Detect events within word (relative times)
                events = sami.detect_all(word_audio, 0)
                events = events[(events >= 0) & (events <= word_duration_sec)]
                
                # Get boundaries
                n_letters = len(word_letters)
                boundaries = katib.dtw_align(n_letters, events, word_duration_sec)
                
                # Apply to letters
                word_letters = muqri.redistribute_with_weights(
                    word_letters, boundaries, word_start_sec, unit_scale
                )
                wave_count += 1
        
        all_letters.extend(word_letters)
    
    # Merge diacritics
    if merge_diacr:
        all_letters = merge_diacritics(all_letters)
    
    # Re-index and clean
    for i, letter in enumerate(all_letters):
        letter['idx'] = i
        # Remove helper fields
        for key in ['charIdx', 'weight']:
            if key in letter:
                del letter[key]
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ {len(all_letters)} letters ({unit_name}), {wave_count} words with wave")
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch Lisan Wave STT v2.0')
    parser.add_argument('--reciter', default='abdul_basit', choices=['abdul_basit', 'mah', 'all'])
    parser.add_argument('--surahs', default='all', help='Comma-separated or "all"')
    parser.add_argument('--no-merge', action='store_true')
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "public/data"
    AUDIO_DIR = PROJECT_ROOT / "public/audio"
    
    print("=" * 60)
    print("Batch Lisan Wave STT v2.0")
    print("=" * 60)
    
    processed = 0
    errors = 0
    
    if args.reciter in ['abdul_basit', 'all']:
        print("\n=== Abdul Basit ===")
        orig_dir = DATA_DIR / "abdul_basit_original"
        out_dir = DATA_DIR / "abdul_basit"
        audio_dir = AUDIO_DIR / "abdul_basit"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if orig_dir.exists():
            for f in sorted(orig_dir.glob("letter_timing_*.json")):
                name = f.stem.replace("letter_timing_", "")
                if name.isdigit():
                    surah_num = int(name)
                    if args.surahs != 'all':
                        requested = [int(s) for s in args.surahs.split(',')]
                        if surah_num not in requested:
                            continue
                    
                    audio_path = audio_dir / f"surah_{surah_num:03d}.mp3"
                    wave_str = " [+wave]" if audio_path.exists() else ""
                    print(f"  Surah {surah_num}...{wave_str}")
                    
                    if process_surah(f, out_dir / f.name, audio_path if audio_path.exists() else None, not args.no_merge):
                        processed += 1
                    else:
                        errors += 1
    
    print(f"\n=== Complete ===")
    print(f"Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
