#!/usr/bin/env python3
"""
WhisperX Local Processing for Surah 1 (CPU Version)
Downloads the QDC audio (matching the app) and generates timing data locally.
"""
import os
import sys
import json
import torch
import whisperx
import urllib.request
from pathlib import Path
from difflib import SequenceMatcher

# Monkeypatch torch.load to allow unsafe globals (needed for WhisperX/OmegaConf in pytorch 2.6+)
# Solution 1: Try to add safe globals
try:
    from omegaconf import OmegaConf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
    print("Added OmegaConf to torch safe globals.")
except ImportError:
    print("OmegaConf not found directly, falling back to aggressive patch.")

# Solution 2: Aggressive patch for torch.load
original_load = torch.load
def safe_load(*args, **kwargs):
    # FORCE weights_only=False even if the caller set it to True
    # This prevents libraries efficiently setting it to True from breaking
    kwargs['weights_only'] = False 
    return original_load(*args, **kwargs)
torch.load = safe_load

# Configuration
SURAH_NUM = 1
AUDIO_URL = "https://download.quranicaudio.com/qdc/abdul_baset/mujawwad/1.mp3"
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "public/audio/abdul_basit"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
DEVICE = "cpu" 
COMPUTE_TYPE = "int8" # Optimized for CPU

# Ensure directories exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quran Text (Uthmani) for Surah 1
QURAN_TEXT_SURAH_1 = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ مَـٰلِكِ يَوْمِ ٱلدِّينِ إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّالِّينَ"

# Phonetic weights
PHONETIC_WEIGHTS = {
    'ا': 1.5, 'آ': 1.5, 'أ': 1.2, 'إ': 1.2, 'ء': 0.8, 'ٱ': 1.2,
    'ب': 1.0, 'ت': 1.0, 'ث': 1.0, 'ج': 1.0, 'ح': 1.0,
    'خ': 1.0, 'د': 1.0, 'ذ': 1.0, 'ر': 1.0, 'ز': 1.0,
    'س': 1.0, 'ش': 1.0, 'ص': 1.0, 'ض': 1.0, 'ط': 1.0,
    'ظ': 1.0, 'ع': 1.3, 'غ': 1.0, 'ف': 1.0, 'ق': 1.0,
    'ك': 1.0, 'ل': 1.0, 'م': 1.0, 'ن': 1.0, 'ه': 1.0,
    'و': 1.3, 'ي': 1.3, 'ى': 1.3, 'ة': 0.8, 'ئ': 0.9,
    'ُ': 0.3, 'َ': 0.3, 'ِ': 0.3, 'ً': 0.3, 'ٌ': 0.3, 'ٍ': 0.3,
    'ّ': 0.2, 'ْ': 0.2, 'ٰ': 0.5, 'ٓ': 0.3
}

def download_audio():
    audio_path = AUDIO_DIR / "surah_001.mp3"
    if audio_path.exists() and audio_path.stat().st_size > 10000:
        print(f"Audio already exists: {audio_path}")
        return audio_path
    
    print(f"Downloading audio from {AUDIO_URL}...")
    urllib.request.urlretrieve(AUDIO_URL, audio_path)
    print("Download complete.")
    return audio_path

def normalize_arabic(text):
    DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
    text = ''.join(c for c in text if c not in DIACRITICS)
    text = text.replace('آ', 'ا').replace('أ', 'ا').replace('إ', 'ا').replace('ٱ', 'ا')
    return text

def build_letter_groups(text):
    DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
    groups = []
    current = []
    for i, char in enumerate(text):
        if char.isspace():
            if current:
                groups.append(current)
                current = []
        elif char not in DIACRITICS:
            if current:
                groups.append(current)
            current = [(i, char)]
        else:
            if current:
                current.append((i, char))
    if current:
        groups.append(current)
    return groups

def main():
    print("=== WhisperX Local Processing (Surah 1) ===")
    
    # 1. Download Audio
    audio_path = download_audio()
    
    # 2. Load Models
    print("Loading WhisperX model (CPU - large-v3)...")
    # Using 'large-v3' as requested (plenty of RAM available)
    # 'int8' is generally faster on CPU than float32
    model = whisperx.load_model("large-v3", DEVICE, compute_type="int8", language="ar")
    
    print("Loading Alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    
    # 3. Transcribe
    print("Transcribing audio...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=4) # Smaller batch for CPU
    
    # 4. Align
    print("Aligning characters...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=True)
    
    # 5. Extract Timing
    whisper_chars = []
    for seg in result.get("segments", []):
        if "chars" in seg:
            for ch in seg["chars"]:
                if "start" in ch and "end" in ch:
                    whisper_chars.append(ch)
    
    print(f"Got {len(whisper_chars)} timed chars from Whisper")
    
    # 6. Match to Quran Text
    whisper_text = "".join(c["char"] for c in whisper_chars)
    whisper_norm = normalize_arabic(whisper_text)
    
    letter_groups = build_letter_groups(QURAN_TEXT_SURAH_1)
    quran_base_chars = [normalize_arabic(g[0][1]) for g in letter_groups]
    quran_norm = "".join(quran_base_chars)
    
    print(f"Aligning {len(whisper_norm)} detected chars to {len(quran_norm)} Quran letters...")
    
    matcher = SequenceMatcher(None, whisper_norm, quran_norm, autojunk=False)
    blocks = matcher.get_matching_blocks()
    
    output_timing = []
    for i, char in enumerate(QURAN_TEXT_SURAH_1):
        output_timing.append({
            "idx": i,
            "char": char,
            "start": 0,
            "end": 0,
            "duration": 0,
            "is_space": char.isspace()
        })
        
    matched_count = 0
    for block in blocks:
        for offset in range(block.size):
            w_idx = block.a + offset
            q_idx = block.b + offset
            
            if w_idx < len(whisper_chars) and q_idx < len(letter_groups):
                w_char = whisper_chars[w_idx]
                start_ms = int(w_char["start"] * 1000)
                end_ms = int(w_char["end"] * 1000)
                
                group = letter_groups[q_idx]
                group_size = len(group)
                if group_size > 0 and end_ms > start_ms:
                    dur = (end_ms - start_ms) / group_size
                    for j, (char_idx, char) in enumerate(group):
                        c_start = start_ms + (j * dur)
                        c_end = c_start + dur
                        output_timing[char_idx]["start"] = int(c_start)
                        output_timing[char_idx]["end"] = int(c_end)
                        output_timing[char_idx]["duration"] = int(dur)
                matched_count += 1

    print(f"Matched {matched_count}/{len(letter_groups)} letters")
    
    # Filter spaces and save
    final_output = [t for t in output_timing if not t["is_space"]]
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH_NUM}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
