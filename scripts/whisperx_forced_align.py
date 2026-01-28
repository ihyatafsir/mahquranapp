#!/usr/bin/env python3
"""
WhisperX Forced Alignment for Surah 1 (CPU Version)
Uses wav2vec2 to FORCE align the known Quran text to the audio.
This gives perfect letter timing since we provide the exact text upfront.
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
DEVICE = "cpu"

# Ensure directories exist
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quran Text (Uthmani) for Surah 1 - the EXACT text we want to align
QURAN_TEXT_SURAH_1 = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ مَـٰلِكِ يَوْمِ ٱلدِّينِ إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّالِّينَ"

def download_audio():
    audio_path = AUDIO_DIR / "surah_001.mp3"
    if audio_path.exists() and audio_path.stat().st_size > 10000:
        print(f"Audio already exists: {audio_path}")
        return audio_path
    
    print(f"Downloading audio from {AUDIO_URL}...")
    urllib.request.urlretrieve(AUDIO_URL, audio_path)
    print("Download complete.")
    return audio_path

def main():
    print("=== WhisperX FORCED ALIGNMENT for Surah 1 ===")
    print("Using known Quran text for direct wav2vec2 alignment")
    
    # 1. Download Audio
    audio_path = download_audio()
    
    # 2. Load Alignment Model (wav2vec2)
    # NOTE: We skip the Whisper transcription model entirely!
    print("Loading wav2vec2 alignment model (Arabic)...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    print("Alignment model loaded.")
    
    # 3. Load Audio
    print("Loading audio...")
    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # 4. Create "fake" segments from the known Quran text
    # WhisperX's align() function expects segments with 'text', 'start', 'end'
    # We provide the full Quran text as a single segment spanning the entire audio
    print("Creating forced alignment segment from Quran text...")
    segments = [{
        "text": QURAN_TEXT_SURAH_1,
        "start": 0.0,
        "end": audio_duration
    }]
    
    # 5. Force Align
    print("Performing FORCED ALIGNMENT with wav2vec2...")
    result = whisperx.align(
        segments, 
        model_a, 
        metadata, 
        audio, 
        DEVICE, 
        return_char_alignments=True
    )
    
    # 6. Extract character-level timing
    print("Extracting character timings...")
    output_timing = []
    idx = 0
    
    for seg in result.get("segments", []):
        if "chars" in seg:
            for ch in seg["chars"]:
                char = ch.get("char", "")
                start = ch.get("start", 0)
                end = ch.get("end", 0)
                
                # Skip spaces
                if char.isspace():
                    continue
                
                output_timing.append({
                    "idx": idx,
                    "char": char,
                    "start": int(start * 1000),  # ms
                    "end": int(end * 1000),      # ms
                    "duration": int((end - start) * 1000)
                })
                idx += 1
    
    print(f"Got {len(output_timing)} characters with timing")
    
    # 7. Save output
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH_NUM}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_timing, f, ensure_ascii=False, indent=2)
        
    print(f"Saved to {output_path}")
    
    # Print first 10 for verification
    print("\nFirst 10 characters:")
    for e in output_timing[:10]:
        print(f"  {e['idx']}: '{e['char']}' @ {e['start']}ms - {e['end']}ms")

if __name__ == "__main__":
    main()
