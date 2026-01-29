#!/usr/bin/env python3
"""
Lisan Wave STT for MAH Reciter

Runs full CTC + wave detection + diacritic merge on MAH audio files.
Output: public/data/letter_timing_{surah}.json (seconds, merged diacritics)
"""
import json
import numpy as np
from pathlib import Path
import sys

try:
    import librosa
    import torch
    from ctc_forced_aligner import (
        load_audio, load_alignment_model, generate_emissions,
        preprocess_text, get_alignments, get_spans, postprocess_results,
    )
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "public/audio"
DATA_DIR = PROJECT_ROOT / "public/data"
VERSES_PATH = DATA_DIR / "verses_v4.json"

DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')
SHADDA = '\u0651'
MADD_LETTERS = set('اويٱى')
HALQ_LETTERS = set('ءهعحغخ')

LISAN_WEIGHTS = {
    'haraka': 1.0, 'sukun': 0.5, 'shadda': 0.6, 'madd': 2.5,
    'halq': 1.4, 'normal': 1.0,
}


class Sami:
    def __init__(self, sr=16000, hop_length=256):
        self.sr = sr
        self.hop_length = hop_length
    
    def detect_all(self, audio):
        if len(audio) < 512:
            return np.array([])
        try:
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, hop_length=self.hop_length, backtrack=True, units='time')
            S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
            flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
            flux = np.concatenate([[0], flux])
            flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)
            threshold = np.mean(flux) + 0.5 * np.std(flux)
            peaks = np.where(flux > threshold)[0]
            local_max = [peaks[i] for i in range(1, len(peaks)-1) if flux[peaks[i]] >= flux[peaks[i]-1] and flux[peaks[i]] >= flux[peaks[i]+1]]
            flux_times = librosa.frames_to_time(np.array(local_max), sr=self.sr, hop_length=self.hop_length)
            all_times = np.concatenate([onsets, flux_times])
            return np.unique(np.sort(all_times))
        except:
            return np.array([])


class Katib:
    def get_letter_units(self, word_text):
        units = []
        i = 0
        while i < len(word_text):
            char = word_text[i]
            if char in DIACRITICS:
                if units: units[-1]['diacritics'] += char
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
        n = len(units)
        if n == 0: return []
        
        duration = word_end - word_start
        word_events = [e - word_start for e in events if word_start <= e <= word_end]
        boundaries = sorted(set([0.0] + word_events + [duration]))
        
        if len(boundaries) != n + 1:
            if len(boundaries) < n + 1:
                boundaries = np.linspace(0, duration, n + 1).tolist()
            else:
                indices = np.round(np.linspace(0, len(boundaries) - 1, n + 1)).astype(int)
                boundaries = [boundaries[i] for i in indices]
        
        return [{'base': u['base'], 'diacritics': u['diacritics'], 
                 'start': word_start + boundaries[i], 'end': word_start + boundaries[i+1]} 
                for i, u in enumerate(units)]


class Muqri:
    def get_weight(self, char, prev=None):
        if char in DIACRITICS:
            if char == SHADDA: return LISAN_WEIGHTS['shadda']
            elif char == '\u0652': return LISAN_WEIGHTS['sukun']
            return LISAN_WEIGHTS['haraka']
        if char in MADD_LETTERS: return LISAN_WEIGHTS['madd']
        if char in HALQ_LETTERS: return LISAN_WEIGHTS['halq']
        return LISAN_WEIGHTS['normal']
    
    def expand_to_characters(self, word_timings):
        result = []
        for unit in word_timings:
            chars = [unit['base']] + list(unit['diacritics'])
            weights = [self.get_weight(c, chars[i-1] if i > 0 else None) for i, c in enumerate(chars)]
            if SHADDA in unit['diacritics']: weights[0] *= 2.0
            total = sum(weights)
            dur = unit['end'] - unit['start']
            curr = unit['start']
            for c, w in zip(chars, weights):
                d = (w / total) * dur
                result.append({'char': c, 'start': curr, 'end': curr + d})
                curr += d
        return result


def merge_diacritics(letters):
    merged = []
    i = 0
    while i < len(letters):
        curr = letters[i].copy()
        combined, end = curr['char'], curr['end']
        j = i + 1
        while j < len(letters) and letters[j]['char'] in DIACRITICS:
            combined += letters[j]['char']
            end = letters[j]['end']
            j += 1
        curr['char'], curr['end'] = combined, end
        merged.append(curr)
        i = j
    return merged


def get_quran_text(surah_num):
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)


def process_surah(surah_num, device="cpu"):
    audio_path = AUDIO_DIR / f"surah_{surah_num:03d}.mp3"
    output_path = DATA_DIR / f"letter_timing_{surah_num}.json"
    
    if not audio_path.exists():
        print(f"  No audio file")
        return False
    
    text = get_quran_text(surah_num)
    if not text:
        print(f"  No Quran text")
        return False
    
    # Load audio
    audio, sr = librosa.load(str(audio_path), sr=16000)
    print(f"  Audio: {len(audio)/sr:.0f}s")
    
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
    sami, katib, muqri = Sami(sr=sr), Katib(), Muqri()
    all_events = sami.detect_all(audio)
    print(f"  Detected {len(all_events)} acoustic events")
    
    # Align
    all_timings = []
    for wt in words:
        if not wt['text'].strip(): continue
        word_events = all_events[(all_events >= wt['start']) & (all_events <= wt['end'])]
        unit_timings = katib.align_word(wt['text'], wt['start'], wt['end'], word_events)
        all_timings.extend(muqri.expand_to_characters(unit_timings))
    
    # Merge and save
    merged = merge_diacritics(all_timings)
    for i, t in enumerate(merged): t['idx'] = i
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(merged)} letters")
    return True


def main():
    # MAH surahs with audio
    MAH_SURAHS = [1, 2, 18, 36, 47, 53, 55, 56, 67, 71, 75, 80, 82, 85, 87, 89, 90, 91, 92, 93, 109, 112, 113, 114]
    
    if len(sys.argv) > 1:
        MAH_SURAHS = [int(s) for s in sys.argv[1:]]
    
    print("=" * 60)
    print("MAH Lisan Wave STT")
    print(f"Processing {len(MAH_SURAHS)} surahs")
    print("=" * 60)
    
    success, failed = 0, 0
    for surah in MAH_SURAHS:
        print(f"\n[Surah {surah}]")
        try:
            if process_surah(surah):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
