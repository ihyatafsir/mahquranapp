#!/usr/bin/env python3
"""
Batch Lisan Wave STT - Full Pipeline for Abdul Basit

Two processing modes:
1. Surahs 2-7: Full CTC + Wave detection (like lisan_wave_stt.py)
2. Surahs 8-114: Use existing word boundaries + Wave refinement

Produces output matching GitHub format (seconds, merged diacritics).
"""
import json
import numpy as np
from pathlib import Path
import sys
import urllib.request

# Import CTC and audio processing
try:
    import librosa
    import torch
    from ctc_forced_aligner import (
        load_audio, load_alignment_model, generate_emissions,
        preprocess_text, get_alignments, get_spans, postprocess_results,
    )
    from scipy.ndimage import gaussian_filter1d
    CTC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    CTC_AVAILABLE = False

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "public/data"
AUDIO_DIR = PROJECT_ROOT / "public/audio/abdul_basit"
OUTPUT_DIR = DATA_DIR / "abdul_basit"
VERSES_PATH = DATA_DIR / "verses_v4.json"

# Diacritics
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'

# Lisan weights
LISAN_WEIGHTS = {
    'haraka': 1.0, 'sukun': 0.5, 'shadda': 0.6, 'madd': 2.5,
    'halq': 1.4, 'normal': 1.0,
}
MADD_LETTERS = set('اويٱى')
HALQ_LETTERS = set('ءهعحغخ')


class Sami:
    """سامع (Listener) - Acoustic event detector."""
    
    def __init__(self, sr=16000, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_all(self, audio):
        """Detect acoustic events (fused onsets + flux)."""
        if len(audio) < 512:
            return np.array([])
        try:
            # Onsets
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, hop_length=self.hop_length, backtrack=True, units='time')
            
            # Spectral flux
            S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            flux = np.concatenate([[0], flux])
            flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)
            threshold = np.mean(flux) + 0.5 * np.std(flux)
            peaks = np.where(flux > threshold)[0]
            local_max = [peaks[i] for i in range(1, len(peaks)-1) if flux[peaks[i]] >= flux[peaks[i]-1] and flux[peaks[i]] >= flux[peaks[i]+1]]
            flux_times = librosa.frames_to_time(np.array(local_max), sr=self.sr, hop_length=self.hop_length)
            
            # Fuse
            all_times = np.concatenate([onsets, flux_times])
            all_times = np.unique(all_times)
            all_times.sort()
            return all_times
        except Exception:
            return np.array([])


class Katib:
    """كاتب (Scribe) - Maps events to letter boundaries."""
    
    def get_letter_units(self, word_text):
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
    
    def align_word(self, word_text, word_start, word_end, events):
        units = self.get_letter_units(word_text)
        n_letters = len(units)
        if n_letters == 0:
            return []
        
        duration = word_end - word_start
        word_events = [e - word_start for e in events if word_start <= e <= word_end]
        
        # Build boundaries
        boundaries = [0.0] + word_events + [duration]
        boundaries = sorted(set(boundaries))
        
        if len(boundaries) != n_letters + 1:
            if len(boundaries) < n_letters + 1:
                boundaries = np.linspace(0, duration, n_letters + 1).tolist()
            else:
                indices = np.round(np.linspace(0, len(boundaries) - 1, n_letters + 1)).astype(int)
                boundaries = [boundaries[i] for i in indices]
        
        timings = []
        for i, unit in enumerate(units):
            timings.append({
                'base': unit['base'],
                'diacritics': unit['diacritics'],
                'start': word_start + boundaries[i],
                'end': word_start + boundaries[i + 1]
            })
        return timings


class Muqri:
    """مقرئ (Reciter) - Applies tajweed timing and expands to characters."""
    
    def get_weight(self, char, prev_char=None):
        if char in DIACRITICS:
            if char == SHADDA: return LISAN_WEIGHTS['shadda']
            elif char == '\u0652': return LISAN_WEIGHTS['sukun']
            return LISAN_WEIGHTS['haraka']
        if char in MADD_LETTERS: return LISAN_WEIGHTS['madd']
        if char in HALQ_LETTERS: return LISAN_WEIGHTS['halq']
        return LISAN_WEIGHTS['normal']
    
    def expand_to_characters(self, word_timings):
        char_timings = []
        for unit in word_timings:
            base = unit['base']
            diacs = unit['diacritics']
            start = unit['start']
            end = unit['end']
            duration = end - start
            
            all_chars = [base] + list(diacs)
            weights = [self.get_weight(c, all_chars[i-1] if i > 0 else None) for i, c in enumerate(all_chars)]
            if SHADDA in diacs:
                weights[0] *= 2.0
            total_weight = sum(weights)
            
            current = start
            for c, w in zip(all_chars, weights):
                char_dur = (w / total_weight) * duration
                char_timings.append({'char': c, 'start': current, 'end': current + char_dur})
                current += char_dur
        return char_timings


def merge_diacritics(letters):
    """Merge diacritics with base letters for unified highlighting."""
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
        merged.append(current)
        i = j
    return merged


def get_quran_text(surah_num):
    """Get Quran text from verses_v4.json."""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)


def process_surah_full_ctc(surah_num, audio_path, text, device="cpu"):
    """Full CTC + Wave detection pipeline (for surahs 2-7)."""
    print(f"  Loading audio...")
    audio, sr = librosa.load(str(audio_path), sr=16000)
    duration = len(audio) / sr
    print(f"  Audio: {duration:.1f}s")
    
    # CTC word boundaries
    print(f"  Getting CTC word boundaries...")
    model, tokenizer = load_alignment_model(device, dtype=torch.float32)
    waveform = load_audio(str(audio_path), model.dtype, model.device)
    emissions, stride = generate_emissions(model, waveform, batch_size=8)
    tokens, text_starred = preprocess_text(text, romanize=True, language='ara')
    segments, scores, blank = get_alignments(emissions, tokens, tokenizer)
    spans = get_spans(tokens, segments, blank)
    words = postprocess_results(text_starred, spans, stride, scores)
    print(f"  Got {len(words)} word boundaries")
    
    # Wave detection
    print(f"  Detecting acoustic events...")
    sami = Sami(sr=sr)
    katib = Katib()
    muqri = Muqri()
    all_events = sami.detect_all(audio)
    print(f"  Detected {len(all_events)} events")
    
    # Align each word
    all_timings = []
    for wt in words:
        word_text = wt['text']
        word_start = wt['start']
        word_end = wt['end']
        if not word_text.strip():
            continue
        word_events = all_events[(all_events >= word_start) & (all_events <= word_end)]
        unit_timings = katib.align_word(word_text, word_start, word_end, word_events)
        char_timings = muqri.expand_to_characters(unit_timings)
        all_timings.extend(char_timings)
    
    # Merge diacritics
    merged = merge_diacritics(all_timings)
    
    # Add indices
    for i, t in enumerate(merged):
        t['idx'] = i
    
    return merged


def process_surah_existing_timing(surah_num, audio_path):
    """Use existing timing data + wave refinement (for surahs 8-114)."""
    orig_path = DATA_DIR / "abdul_basit_original" / f"letter_timing_{surah_num}.json"
    if not orig_path.exists():
        print(f"  No original timing data found")
        return None
    
    with open(orig_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        return None
    
    # Detect units (ms vs seconds)
    unit_scale = 1000 if data[0].get('start', 0) > 100 else 1
    
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=16000)
    sami = Sami(sr=sr)
    katib = Katib()
    
    # Inject wordIdx if missing
    if 'wordIdx' not in data[0]:
        gap_threshold = 50 if unit_scale == 1000 else 0.05
        word_idx = 0
        data[0]['wordIdx'] = word_idx
        for i in range(1, len(data)):
            gap = data[i].get('start', 0) - data[i-1].get('end', 0)
            if gap > gap_threshold:
                word_idx += 1
            data[i]['wordIdx'] = word_idx
    
    # Group by word
    words = {}
    for letter in data:
        idx = letter['wordIdx']
        if idx not in words:
            words[idx] = []
        words[idx].append(letter)
    
    # Process each word with wave detection
    all_letters = []
    for word_idx in sorted(words.keys()):
        word_letters = words[word_idx]
        word_start_sec = word_letters[0]['start'] / unit_scale
        word_end_sec = word_letters[-1]['end'] / unit_scale
        
        # Get word audio and events
        start_sample = int(word_start_sec * sr)
        end_sample = int(word_end_sec * sr)
        if 0 <= start_sample < end_sample <= len(audio):
            word_audio = audio[start_sample:end_sample]
            events = sami.detect_all(word_audio)
            # Events are relative to word, convert to absolute seconds
            events = events + word_start_sec
            
            # Align characters within word
            word_text = ''.join(l['char'] for l in word_letters)
            unit_timings = katib.align_word(word_text, word_start_sec, word_end_sec, events)
            
            for ut in unit_timings:
                all_letters.append({
                    'char': ut['base'] + ut['diacritics'],
                    'start': ut['start'],
                    'end': ut['end']
                })
        else:
            # Keep original timing (converted to seconds)
            for l in word_letters:
                all_letters.append({
                    'char': l['char'],
                    'start': l['start'] / unit_scale,
                    'end': l['end'] / unit_scale
                })
    
    # Add indices
    for i, l in enumerate(all_letters):
        l['idx'] = i
    
    return all_letters


def main():
    start_surah = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    end_surah = int(sys.argv[2]) if len(sys.argv) > 2 else 114
    
    print("=" * 60)
    print("Batch Lisan Wave STT - Full Pipeline")
    print(f"Processing surahs {start_surah}-{end_surah}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    
    for surah in range(start_surah, end_surah + 1):
        audio_path = AUDIO_DIR / f"surah_{surah:03d}.mp3"
        output_path = OUTPUT_DIR / f"letter_timing_{surah}.json"
        
        if not audio_path.exists():
            print(f"\n[Surah {surah}] No audio file - skipping")
            failed += 1
            continue
        
        print(f"\n[Surah {surah}]")
        
        try:
            if surah <= 7:
                # Full CTC pipeline for surahs 2-7
                text = get_quran_text(surah)
                if not text:
                    print(f"  No Quran text found")
                    failed += 1
                    continue
                print(f"  Mode: Full CTC + Wave")
                result = process_surah_full_ctc(surah, audio_path, text)
            else:
                # Existing timing + wave for surahs 8+
                print(f"  Mode: Existing timing + Wave")
                result = process_surah_existing_timing(surah, audio_path)
            
            if result:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Saved {len(result)} letters")
                success += 1
            else:
                print(f"  ✗ No result")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
