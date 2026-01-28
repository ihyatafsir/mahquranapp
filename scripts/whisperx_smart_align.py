#!/usr/bin/env python3
"""
Smarter WhisperX Alignment for Surah 1 (CPU Version)

Strategy:
1. Strip diacritics from Quran text before alignment (wav2vec only hears base letters)
2. Force align base-letter-only text to audio with wav2vec2
3. Expand timing back to include diacritics by grouping each base letter with its following diacritics

This gives accurate timing for base letters while diacritics inherit their base letter's timing.
"""
import os
import sys
import json
import torch
import whisperx
import urllib.request
from pathlib import Path

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
try:
    from omegaconf import OmegaConf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
    print("Added OmegaConf to torch safe globals.")
except ImportError:
    print("OmegaConf not found, using aggressive torch.load patch.")

original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False 
    return original_load(*args, **kwargs)
torch.load = safe_load

# Configuration
SURAH_NUM = 1
AUDIO_URL = "https://download.quranicaudio.com/qdc/abdul_baset/mujawwad/1.mp3"
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "public/audio/abdul_basit"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
DEVICE = "cpu"

# Ensure directories exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arabic diacritics (tashkeel)
DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def load_quran_text(surah_num: int) -> str:
    """Load Quran text from verses_v4.json (same source as App.tsx)"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    
    surah_verses = all_verses.get(str(surah_num), [])
    # Join all verse texts with space
    full_text = ' '.join(v.get('text', '') for v in surah_verses)
    return full_text

def download_audio():
    audio_path = AUDIO_DIR / "surah_001.mp3"
    if audio_path.exists() and audio_path.stat().st_size > 10000:
        print(f"Audio already exists: {audio_path}")
        return audio_path
    
    print(f"Downloading audio from {AUDIO_URL}...")
    urllib.request.urlretrieve(AUDIO_URL, audio_path)
    print("Download complete.")
    return audio_path

def strip_diacritics(text: str) -> str:
    """Remove all diacritics from text"""
    return ''.join(c for c in text if c not in DIACRITICS)

def build_letter_groups(text: str) -> list:
    """
    Group each base letter with its following diacritics.
    Returns list of groups, each group is [(original_idx, char), ...]
    """
    groups = []
    current_group = []
    
    for i, char in enumerate(text):
        if char.isspace():
            if current_group:
                groups.append(current_group)
                current_group = []
            # Skip spaces
        elif char in DIACRITICS:
            # Diacritic: add to current group
            if current_group:
                current_group.append((i, char))
        else:
            # Base letter: start new group
            if current_group:
                groups.append(current_group)
            current_group = [(i, char)]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def main():
    print("=== Smarter WhisperX Alignment for Surah 1 ===")
    print("Strategy: Strip diacritics -> Align -> Expand timing")
    
    # 1. Download Audio
    audio_path = download_audio()
    
    # 2. Load Quran text from verses_v4.json (same source App.tsx uses)
    print("Loading Quran text from verses_v4.json...")
    quran_text = load_quran_text(SURAH_NUM)
    print(f"  Loaded text: {len(quran_text)} chars (with spaces)")
    no_spaces = quran_text.replace(' ', '')
    print(f"  Without spaces: {len(no_spaces)} chars")
    
    # 3. Build letter groups from full Quran text
    print("Building letter groups (base letter + diacritics)...")
    letter_groups = build_letter_groups(quran_text)
    print(f"  Found {len(letter_groups)} letter groups")
    
    # 4. Create base-letter-only text for alignment
    base_letters_only = strip_diacritics(quran_text).replace(' ', '')  # also strip spaces
    print(f"  Base letters only: {len(base_letters_only)} chars")
    print(f"  First 20: {base_letters_only[:20]}")
    
    # 4. Load Alignment Model (wav2vec2)
    print("Loading wav2vec2 alignment model (Arabic)...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    print("Alignment model loaded.")
    
    # 5. Load Audio
    print("Loading audio...")
    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # 6. Create segment with base-letters-only text
    # Put spaces between each base letter to help alignment? 
    # No, let's just use the base letters concatenated without spaces
    # Actually, let's use a space-separated version for better phoneme boundary detection
    # Hmm, WhisperX align expects natural text. Let's try adding spaces back
    
    # Re-create with spaces preserved but diacritics stripped
    base_with_spaces = strip_diacritics(quran_text)
    print(f"  Base with spaces: '{base_with_spaces[:50]}...'")
    
    segments = [{
        "text": base_with_spaces,
        "start": 0.0,
        "end": audio_duration
    }]
    
    # 7. Force Align
    print("Performing FORCED ALIGNMENT with wav2vec2...")
    result = whisperx.align(
        segments, 
        model_a, 
        metadata, 
        audio, 
        DEVICE, 
        return_char_alignments=True
    )
    
    # 8. Extract character-level timing for base letters
    print("Extracting base letter timings...")
    base_timings = []
    
    for seg in result.get("segments", []):
        if "chars" in seg:
            for ch in seg["chars"]:
                char = ch.get("char", "")
                start = ch.get("start", 0)
                end = ch.get("end", 0)
                
                # Skip spaces
                if char.isspace():
                    continue
                
                base_timings.append({
                    "char": char,
                    "start": start,
                    "end": end
                })
    
    print(f"Got {len(base_timings)} base letter timings")
    
    # 9. Expand timing to include diacritics
    # Match base_timings to letter_groups
    print("Expanding timing to include diacritics...")
    
    output_timing = []
    
    if len(base_timings) != len(letter_groups):
        print(f"  WARNING: Timing count ({len(base_timings)}) != Group count ({len(letter_groups)})")
        print(f"  Will align what we can...")
    
    for gIdx, group in enumerate(letter_groups):
        if gIdx < len(base_timings):
            bt = base_timings[gIdx]
            start_ms = int(bt["start"] * 1000)
            end_ms = int(bt["end"] * 1000)
            duration = end_ms - start_ms
            
            # Distribute duration across all chars in group
            group_size = len(group)
            char_duration = duration / group_size if group_size > 0 else 0
            
            for cIdx, (orig_idx, char) in enumerate(group):
                char_start = start_ms + (cIdx * char_duration)
                char_end = char_start + char_duration
                
                output_timing.append({
                    "idx": len(output_timing),
                    "char": char,
                    "start": int(char_start),
                    "end": int(char_end),
                    "duration": int(char_duration),
                    "groupIdx": gIdx  # Track which group this belongs to
                })
        else:
            # No timing data for this group - use last known end time
            last_end = output_timing[-1]["end"] if output_timing else 0
            for cIdx, (orig_idx, char) in enumerate(group):
                output_timing.append({
                    "idx": len(output_timing),
                    "char": char,
                    "start": last_end,
                    "end": last_end,
                    "duration": 0,
                    "groupIdx": gIdx
                })
    
    print(f"Final output: {len(output_timing)} characters with timing")
    
    # 10. Save output
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH_NUM}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_timing, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {output_path}")
    
    # Print first 15 for verification
    print("\nFirst 15 characters (with diacritics):")
    for e in output_timing[:15]:
        print(f"  {e['idx']}: '{e['char']}' @ {e['start']}ms - {e['end']}ms (group {e['groupIdx']})")

if __name__ == "__main__":
    main()
