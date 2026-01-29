#!/usr/bin/env python3
"""
Batch Lisan Aligner v1.1 (With Wave Detection)

Uses EXISTING word boundaries from timing data and applies:
1. Lisan Tajweed ratios for letter distribution
2. Wave detection (spectral flux) for acoustic boundary refinement
3. Madd energy analysis for elongation detection
4. Phonetic profiling for text-informed peak picking

Requires audio files for full accuracy.
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
LONG_VOWELS = set('ٰ')
HALQ_LETTERS = set('ءهعحغخ')
QALQALAH = set('قطبجد')
GHUNNA_LETTERS = set('نم')

# Lisan Tajweed Ratios
LISAN_RATIOS = {
    'consonant': 1.0,
    'kasra': 0.6,
    'fatha': 0.55,
    'damma': 0.55,
    'shadda': 0.25,
    'sukun': 0.2,
    'tanween': 0.4,
    'madd_asli': 1.5,
    'madd_muttasil': 2.0,
    'madd_lazim': 3.0,
    'halq': 1.2,
    'qalqalah': 1.1,
    'ghunna': 1.3,
}


def get_lisan_ratio(char, prev_char=None, next_char=None):
    """Get Lisan-based ratio with Tajweed coverage."""
    if char in DIACRITICS:
        if char == SHADDA:
            return LISAN_RATIOS['shadda']
        elif char == '\u0652':
            return LISAN_RATIOS['sukun']
        elif char == '\u0650':
            return LISAN_RATIOS['kasra']
        elif char == '\u064E':
            return LISAN_RATIOS['fatha']
        elif char == '\u064F':
            return LISAN_RATIOS['damma']
        elif char in '\u064B\u064C\u064D':
            return LISAN_RATIOS['tanween']
        else:
            return 0.3
    
    if char in GHUNNA_LETTERS:
        if next_char == SHADDA:
            return LISAN_RATIOS['ghunna']
        return 1.0
    
    if char in QALQALAH:
        if next_char == '\u0652' or next_char is None:
            return LISAN_RATIOS['qalqalah']
        return 1.0
    
    if char in MADD_LETTERS or char in LONG_VOWELS:
        if next_char == SHADDA or next_char == '\u0652':
            return LISAN_RATIOS['madd_lazim']
        if next_char == 'ء':
            return LISAN_RATIOS['madd_muttasil']
        if prev_char:
            if char == 'ا' and prev_char == '\u064E':
                return LISAN_RATIOS['madd_asli']
            elif char in 'وٱ' and prev_char == '\u064F':
                return LISAN_RATIOS['madd_asli']
            elif char == 'ي' and prev_char == '\u0650':
                return LISAN_RATIOS['madd_asli']
        return 1.4
    
    if char in HALQ_LETTERS:
        return LISAN_RATIOS['halq']
    
    return LISAN_RATIOS['consonant']


# Phonetic Acoustic Profiles (Text-Informed Detection)
PHONETIC_PROFILES = {
    'stop_consonants': set('بتطدكقء'),
    'nasals': set('من'),
    'fricatives': set('سشصزظذثف'),
    'liquids': set('لر'),
    'vowels': set('اوي'),
    'throat': set('عحغخه'),
}


def get_acoustic_profile(char):
    """Get expected acoustic behavior for a letter."""
    if char in PHONETIC_PROFILES['stop_consonants']:
        return 'onset'
    elif char in PHONETIC_PROFILES['vowels'] or char in DIACRITICS:
        return 'sustain'
    elif char in PHONETIC_PROFILES['nasals']:
        return 'dip'
    elif char in PHONETIC_PROFILES['fricatives']:
        return 'noise'
    return 'generic'


def detect_madd_from_wave(segment, char, sr=16000):
    """Detect if a madd letter is elongated by measuring energy stability."""
    if not WAVE_DETECTION_AVAILABLE:
        return 1.0
    if char not in MADD_LETTERS and char not in LONG_VOWELS:
        return 1.0
    if len(segment) < 256:
        return 1.0
    
    hop = 128
    energy = librosa.feature.rms(y=segment, hop_length=hop)[0]
    if len(energy) < 3:
        return 1.0
    
    mean_e = np.mean(energy)
    if mean_e == 0:
        return 1.0
    cv = np.std(energy) / mean_e
    
    if cv < 0.3:
        return 2.0  # Strong madd
    elif cv < 0.5:
        return 1.5  # Light madd
    else:
        return 1.2


def detect_acoustic_boundaries(segment, n_letters, chars, sr=16000):
    """Detect acoustic boundaries inside a word using spectral flux."""
    if not WAVE_DETECTION_AVAILABLE:
        return np.linspace(0, len(segment)/sr, n_letters + 1)
    if len(segment) < 256 or n_letters <= 1:
        return np.linspace(0, len(segment)/sr, n_letters + 1)
    
    # Spectral flux
    S = np.abs(librosa.stft(segment, hop_length=128))
    flux = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max))
    flux = gaussian_filter1d(flux, sigma=1)
    
    # Peak picking
    peaks = librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=1)
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=128)
    
    duration = len(segment) / sr
    boundaries = np.concatenate([[0], peak_times, [duration]])
    boundaries = np.sort(np.unique(boundaries))
    
    if len(boundaries) != n_letters + 1:
        return np.linspace(0, duration, n_letters + 1)
    
    return boundaries


def redistribute_word(letters, audio_segment=None, sr=16000, time_divisor=1):
    """
    Redistribute letter durations within a word using Lisan ratios.
    If audio_segment provided, uses wave detection for refinement.
    time_divisor: 1000 if original times are in ms, 1 if in seconds.
    """
    if not letters:
        return letters
    
    word_start = letters[0]['start']
    word_end = letters[-1]['end']
    word_duration = word_end - word_start
    
    if word_duration <= 0:
        return letters
    
    # Calculate ratios
    chars = [l['char'] for l in letters]
    n_chars = len(chars)
    ratios = []
    
    for i, c in enumerate(chars):
        prev_c = chars[i-1] if i > 0 else None
        next_c = chars[i+1] if i < len(chars)-1 else None
        ratio = get_lisan_ratio(c, prev_c, next_c)
        
        # Wave-based Madd detection if audio available
        if audio_segment is not None and WAVE_DETECTION_AVAILABLE:
            if c in MADD_LETTERS or c in LONG_VOWELS:
                # Estimate char position in audio
                est_start = int((i / n_chars) * len(audio_segment))
                est_end = int(((i + 1) / n_chars) * len(audio_segment))
                if est_end > est_start:
                    wave_mult = detect_madd_from_wave(audio_segment[est_start:est_end], c, sr)
                    ratio *= wave_mult
        
        ratios.append(ratio)
    
    # Normalize
    total_ratio = sum(ratios)
    if total_ratio == 0:
        total_ratio = 1
    norm_ratios = [r / total_ratio for r in ratios]
    
    # If audio available, try acoustic boundary snapping
    if audio_segment is not None and WAVE_DETECTION_AVAILABLE and n_chars > 1:
        # acoustic_bounds are in seconds, scale to original units
        acoustic_bounds_sec = detect_acoustic_boundaries(audio_segment, n_chars, chars, sr)
        acoustic_bounds = acoustic_bounds_sec * time_divisor  # Convert to original units
        
        # Calculate expected boundaries from ratios
        cum_ratios = np.cumsum([0] + norm_ratios)
        expected_boundaries = cum_ratios * word_duration
        
        # Smart blend: snap to acoustic if close
        final_boundaries = [0]
        for k in range(1, n_chars):
            expected = expected_boundaries[k]
            candidates = acoustic_bounds[1:-1]
            if len(candidates) > 0:
                nearest_idx = (np.abs(candidates - expected)).argmin()
                detected = candidates[nearest_idx]
                # Snap if within 30% of word duration
                if abs(detected - expected) < 0.3 * word_duration:
                    final_boundaries.append(detected)
                else:
                    final_boundaries.append(expected)
            else:
                final_boundaries.append(expected)
        final_boundaries.append(word_duration)
        
        # Apply boundaries
        for i, letter in enumerate(letters):
            letter['start'] = word_start + final_boundaries[i]
            letter['end'] = word_start + final_boundaries[i + 1]
            letter['duration'] = letter['end'] - letter['start']
    else:
        # Pure ratio redistribution (no audio)
        current = word_start
        for i, (letter, ratio) in enumerate(zip(letters, norm_ratios)):
            new_duration = ratio * word_duration
            letter['start'] = current
            letter['end'] = current + new_duration
            letter['duration'] = new_duration
            current += new_duration
    
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
            if next_l['char'] in DIACRITICS or next_l['char'] == SHADDA:
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


def process_timing_file(input_path, output_path, audio_path=None, merge_diacr=True):
    """Process a single timing file with Lisan redistribution + wave detection."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        print(f"  Skipping {input_path} - invalid format")
        return False
    
    # Check if wordIdx exists
    if 'wordIdx' not in data[0]:
        print(f"  Skipping {input_path} - no wordIdx field")
        return False
    
    # Load audio if available
    audio = None
    sr = 16000
    if audio_path and Path(audio_path).exists() and WAVE_DETECTION_AVAILABLE:
        try:
            audio, sr = librosa.load(str(audio_path), sr=sr)
        except Exception as e:
            print(f"  Warning: Could not load audio {audio_path}: {e}")
            audio = None
    
    # Detect if times are in milliseconds or seconds
    first_start = data[0]['start']
    is_milliseconds = first_start > 100  # If start > 100, assume ms
    time_divisor = 1000 if is_milliseconds else 1
    
    # Group by word
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
        
        # Extract audio segment for this word if audio available
        audio_segment = None
        if audio is not None and len(word_letters) > 0:
            # Convert times to seconds using detected divisor
            word_start_sec = word_letters[0]['start'] / time_divisor
            word_end_sec = word_letters[-1]['end'] / time_divisor
            word_start_sample = int(word_start_sec * sr)
            word_end_sample = int(word_end_sec * sr)
            if word_end_sample > word_start_sample and word_end_sample <= len(audio):
                audio_segment = audio[word_start_sample:word_end_sample]
        
        word_letters = redistribute_word(word_letters, audio_segment, sr, time_divisor)
        all_letters.extend(word_letters)
    
    # Merge diacritics if requested
    if merge_diacr:
        all_letters = merge_diacritics(all_letters)
    
    # Re-index
    for i, letter in enumerate(all_letters):
        letter['idx'] = i
        if 'charIdx' in letter:
            del letter['charIdx']
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch Lisan Aligner v1.1 (With Wave Detection)')
    parser.add_argument('--reciter', type=str, default='mah', 
                        choices=['mah', 'abdul_basit', 'all'],
                        help='Reciter to process')
    parser.add_argument('--surahs', type=str, default='all',
                        help='Comma-separated surah numbers or "all"')
    parser.add_argument('--no-merge', action='store_true',
                        help='Do not merge diacritics')
    parser.add_argument('--no-wave', action='store_true',
                        help='Disable wave detection (use ratios only)')
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "public/data"
    AUDIO_DIR = PROJECT_ROOT / "public/audio"
    
    print("=" * 60)
    print("Batch Lisan Aligner v1.1 (With Wave Detection)")
    print("Using existing word boundaries + Tajweed ratios + Wave analysis")
    print("=" * 60)
    
    if not WAVE_DETECTION_AVAILABLE:
        print("Note: librosa not available. Using ratio-only mode.")
    elif args.no_wave:
        print("Note: Wave detection disabled by --no-wave flag.")
    
    processed = 0
    errors = 0
    wave_processed = 0
    
    # Process MAH
    if args.reciter in ['mah', 'all']:
        print("\n=== Processing MAH ===")
        mah_files = list(DATA_DIR.glob("letter_timing_*.json"))
        mah_files = [f for f in mah_files if not f.name.startswith("letter_timing_n")]
        
        for f in sorted(mah_files):
            name = f.stem.replace("letter_timing_", "")
            if name.isdigit():
                surah_num = int(name)
                if args.surahs != 'all':
                    requested = [int(s) for s in args.surahs.split(',')]
                    if surah_num not in requested:
                        continue
                
                # Find audio file for this surah
                audio_path = None
                if not args.no_wave:
                    audio_file = AUDIO_DIR / f"surah_{surah_num:03d}.mp3"
                    if audio_file.exists():
                        audio_path = audio_file
                        wave_processed += 1
                
                wave_str = " [+wave]" if audio_path else ""
                print(f"  Processing Surah {surah_num}...{wave_str}")
                
                if process_timing_file(f, f, audio_path, not args.no_merge):
                    processed += 1
                else:
                    errors += 1
    
    # Process Abdul Basit
    if args.reciter in ['abdul_basit', 'all']:
        print("\n=== Processing Abdul Basit ===")
        ab_dir = DATA_DIR / "abdul_basit"
        ab_audio_dir = AUDIO_DIR / "abdul_basit"
        
        if ab_dir.exists():
            ab_files = list(ab_dir.glob("letter_timing_*.json"))
            
            for f in sorted(ab_files):
                name = f.stem.replace("letter_timing_", "")
                if name.isdigit():
                    surah_num = int(name)
                    if args.surahs != 'all':
                        requested = [int(s) for s in args.surahs.split(',')]
                        if surah_num not in requested:
                            continue
                    
                    # Find audio file for this surah
                    audio_path = None
                    if not args.no_wave:
                        audio_file = ab_audio_dir / f"surah_{surah_num:03d}.mp3"
                        if audio_file.exists():
                            audio_path = audio_file
                            wave_processed += 1
                    
                    wave_str = " [+wave]" if audio_path else ""
                    print(f"  Processing Surah {surah_num}...{wave_str}")
                    
                    if process_timing_file(f, f, audio_path, not args.no_merge):
                        processed += 1
                    else:
                        errors += 1
    
    print(f"\n=== Complete ===")
    print(f"Processed: {processed} files ({wave_processed} with wave detection)")
    print(f"Errors: {errors} files")


if __name__ == "__main__":
    main()

