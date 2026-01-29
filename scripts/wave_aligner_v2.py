#!/usr/bin/env python3
"""
Wave Aligner v2.0 - Proper Unit Handling

Processes timing data with wave detection, preserving original time units.
Works with both milliseconds and seconds input.

Key principles:
1. Detect input units (ms vs seconds)
2. Work internally in seconds for audio processing
3. Output in original units for compatibility
"""
import json
import numpy as np
from pathlib import Path
import argparse

# Try to import librosa for wave detection
try:
    import librosa
    from scipy.ndimage import gaussian_filter1d
    WAVE_DETECTION_AVAILABLE = True
except ImportError:
    WAVE_DETECTION_AVAILABLE = False
    print("Warning: librosa not available. Wave detection disabled.")

# Arabic character sets
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
MADD_LETTERS = set('اويٱى')

# Lisan Tajweed Ratios
LISAN_RATIOS = {
    'consonant': 1.0, 'kasra': 0.6, 'fatha': 0.55, 'damma': 0.55,
    'shadda': 0.25, 'sukun': 0.2, 'tanween': 0.4,
    'madd_asli': 1.5, 'madd_muttasil': 2.0, 'madd_lazim': 3.0,
    'halq': 1.2, 'qalqalah': 1.1, 'ghunna': 1.3,
}


def inject_word_idx(data, gap_threshold=50):
    """
    Inject wordIdx into data by detecting gaps in timing.
    gap_threshold: minimum gap (in original units) to consider as word boundary.
    """
    if len(data) == 0:
        return data
    
    word_idx = 0
    data[0]['wordIdx'] = word_idx
    
    for i in range(1, len(data)):
        prev_end = data[i-1].get('end', 0)
        curr_start = data[i].get('start', 0)
        gap = curr_start - prev_end
        
        # If gap > threshold, start new word
        if gap > gap_threshold:
            word_idx += 1
        
        data[i]['wordIdx'] = word_idx
    
    return data


def detect_units(data):
    """Detect if timing is in milliseconds or seconds."""
    if len(data) == 0:
        return 1  # Default to seconds
    first_start = data[0].get('start', 0)
    # If first non-zero start > 100, likely milliseconds
    if first_start > 100:
        return 1000  # ms
    # Check end of first letter
    first_end = data[0].get('end', 0)
    if first_end > 100:
        return 1000  # ms
    return 1  # seconds


def get_lisan_ratio(char, prev_char=None, next_char=None):
    """Get Lisan-based ratio for a character."""
    if char in DIACRITICS:
        if char == '\u0651': return LISAN_RATIOS['shadda']
        elif char == '\u0652': return LISAN_RATIOS['sukun']
        elif char == '\u0650': return LISAN_RATIOS['kasra']
        elif char == '\u064E': return LISAN_RATIOS['fatha']
        elif char == '\u064F': return LISAN_RATIOS['damma']
        elif char in '\u064B\u064C\u064D': return LISAN_RATIOS['tanween']
        return 0.3
    if char in MADD_LETTERS:
        return LISAN_RATIOS['madd_asli']
    return LISAN_RATIOS['consonant']


def detect_onsets_in_segment(audio_segment, sr=16000):
    """Detect onset times in audio segment using spectral flux."""
    if not WAVE_DETECTION_AVAILABLE or len(audio_segment) < 512:
        return np.array([])
    
    try:
        # Spectral flux for onset detection
        hop_length = 256
        S = np.abs(librosa.stft(audio_segment, hop_length=hop_length))
        onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max))
        onset_env = gaussian_filter1d(onset_env, sigma=1)
        
        # Dynamic threshold based on signal
        threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
        
        # Peak picking
        peaks = librosa.util.peak_pick(onset_env, pre_max=2, post_max=2, 
                                        pre_avg=3, post_avg=3, 
                                        delta=threshold * 0.5, wait=2)
        
        # Convert to time in seconds
        onset_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        return onset_times
    except Exception:
        return np.array([])


def redistribute_word_with_wave(letters, audio, sr, word_start_sec, word_end_sec, unit_scale):
    """
    Redistribute letters within a word using wave detection + Tajweed ratios.
    
    letters: list of letter dicts
    audio: full audio array
    sr: sample rate
    word_start_sec: word start in seconds
    word_end_sec: word end in seconds
    unit_scale: 1000 for ms, 1 for seconds (for output)
    """
    if len(letters) <= 1:
        return letters
    
    # Extract audio segment for this word
    start_sample = int(word_start_sec * sr)
    end_sample = int(word_end_sec * sr)
    
    if end_sample <= start_sample or end_sample > len(audio):
        # Can't process, return original
        return letters
    
    word_audio = audio[start_sample:end_sample]
    word_duration_sec = word_end_sec - word_start_sec
    
    # Calculate Tajweed ratios for each letter
    chars = [l['char'] for l in letters]
    n_chars = len(chars)
    ratios = []
    for i, c in enumerate(chars):
        prev_c = chars[i-1] if i > 0 else None
        next_c = chars[i+1] if i < len(chars)-1 else None
        ratios.append(get_lisan_ratio(c, prev_c, next_c))
    
    # Normalize ratios
    total = sum(ratios)
    if total == 0:
        total = 1
    norm_ratios = [r / total for r in ratios]
    
    # Calculate expected boundaries from ratios
    cum_ratios = np.cumsum([0] + norm_ratios)
    expected_boundaries_sec = cum_ratios * word_duration_sec
    
    # Detect onsets in the word audio
    onsets = detect_onsets_in_segment(word_audio, sr)
    
    # Build final boundaries - snap to onsets if close
    final_boundaries_sec = [0.0]
    
    for k in range(1, n_chars):
        expected = expected_boundaries_sec[k]
        
        if len(onsets) > 0:
            # Find nearest onset
            nearest_idx = np.argmin(np.abs(onsets - expected))
            detected = onsets[nearest_idx]
            
            # Snap if within 20% of word duration
            if abs(detected - expected) < 0.2 * word_duration_sec:
                final_boundaries_sec.append(detected)
            else:
                final_boundaries_sec.append(expected)
        else:
            final_boundaries_sec.append(expected)
    
    final_boundaries_sec.append(word_duration_sec)
    
    # Apply boundaries to letters (converting to original units)
    for i, letter in enumerate(letters):
        new_start_sec = word_start_sec + final_boundaries_sec[i]
        new_end_sec = word_start_sec + final_boundaries_sec[i + 1]
        
        letter['start'] = new_start_sec * unit_scale
        letter['end'] = new_end_sec * unit_scale
        letter['duration'] = (new_end_sec - new_start_sec) * unit_scale
    
    return letters


def merge_diacritics(letters):
    """Merge diacritics with their base letters."""
    merged = []
    i = 0
    while i < len(letters):
        current = letters[i].copy()
        combined_char = current['char']
        combined_end = current['end']
        
        j = i + 1
        while j < len(letters):
            next_l = letters[j]
            if next_l['char'] in DIACRITICS:
                combined_char += next_l['char']
                combined_end = next_l['end']
                j += 1
            else:
                break
        
        current['char'] = combined_char
        current['end'] = combined_end
        current['duration'] = combined_end - current['start']
        merged.append(current)
        i = j
    
    return merged


def process_file(input_path, output_path, audio_path=None, merge_diacr=True):
    """Process a timing file with wave detection."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        print(f"  Skipping {input_path} - invalid format")
        return False
    
    # Check for wordIdx - inject if missing
    has_word_idx = 'wordIdx' in data[0]
    if not has_word_idx:
        # Detect units first to set appropriate gap threshold
        unit_scale = detect_units(data)
        gap_threshold = 50 if unit_scale == 1000 else 0.05  # 50ms or 0.05s
        data = inject_word_idx(data, gap_threshold)
        print(f"    (injected wordIdx based on timing gaps)")
    
    # Detect units
    unit_scale = detect_units(data)
    unit_name = "ms" if unit_scale == 1000 else "sec"
    
    # Load audio
    audio = None
    sr = 16000
    if audio_path and Path(audio_path).exists() and WAVE_DETECTION_AVAILABLE:
        try:
            audio, sr = librosa.load(str(audio_path), sr=sr)
        except Exception as e:
            print(f"  Warning: Could not load audio: {e}")
            audio = None
    
    # Group letters by word
    words = {}
    for letter in data:
        word_idx = letter['wordIdx']
        if word_idx not in words:
            words[word_idx] = []
        words[word_idx].append(letter)
    
    # Process each word
    all_letters = []
    for word_idx in sorted(words.keys()):
        word_letters = words[word_idx]
        
        if audio is not None and len(word_letters) > 0:
            # Convert word boundaries to seconds
            word_start_sec = word_letters[0]['start'] / unit_scale
            word_end_sec = word_letters[-1]['end'] / unit_scale
            
            word_letters = redistribute_word_with_wave(
                word_letters, audio, sr, 
                word_start_sec, word_end_sec, unit_scale
            )
        
        all_letters.extend(word_letters)
    
    # Merge diacritics
    if merge_diacr:
        all_letters = merge_diacritics(all_letters)
    
    # Re-index
    for i, letter in enumerate(all_letters):
        letter['idx'] = i
        # Remove redundant keys
        for key in ['charIdx', 'is_space']:
            if key in letter:
                del letter[key]
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(all_letters)} letters ({unit_name})")
    return True


def main():
    parser = argparse.ArgumentParser(description='Wave Aligner v2.0')
    parser.add_argument('--reciter', type=str, default='abdul_basit',
                        choices=['mah', 'abdul_basit', 'all'])
    parser.add_argument('--surahs', type=str, default='all')
    parser.add_argument('--no-merge', action='store_true')
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "public/data"
    AUDIO_DIR = PROJECT_ROOT / "public/audio"
    
    print("=" * 60)
    print("Wave Aligner v2.0 - Proper Unit Handling")
    print("=" * 60)
    
    processed = 0
    errors = 0
    
    # Process Abdul Basit from ORIGINAL data
    if args.reciter in ['abdul_basit', 'all']:
        print("\n=== Processing Abdul Basit ===")
        orig_dir = DATA_DIR / "abdul_basit_original"
        out_dir = DATA_DIR / "abdul_basit"
        audio_dir = AUDIO_DIR / "abdul_basit"
        
        if orig_dir.exists():
            timing_files = list(orig_dir.glob("letter_timing_*.json"))
            
            for f in sorted(timing_files):
                name = f.stem.replace("letter_timing_", "")
                if name.isdigit():
                    surah_num = int(name)
                    if args.surahs != 'all':
                        requested = [int(s) for s in args.surahs.split(',')]
                        if surah_num not in requested:
                            continue
                    
                    # Find audio
                    audio_path = audio_dir / f"surah_{surah_num:03d}.mp3"
                    wave_str = " [+wave]" if audio_path.exists() else ""
                    print(f"  Processing Surah {surah_num}...{wave_str}")
                    
                    output_path = out_dir / f.name
                    if process_file(f, output_path, audio_path if audio_path.exists() else None, not args.no_merge):
                        processed += 1
                    else:
                        errors += 1
    
    # Process MAH
    if args.reciter in ['mah', 'all']:
        print("\n=== Processing MAH ===")
        # MAH files in main data folder
        # (would need orig_dir for MAH too if available)
        pass  # TODO: Add MAH support when original data available
    
    print(f"\n=== Complete ===")
    print(f"Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
