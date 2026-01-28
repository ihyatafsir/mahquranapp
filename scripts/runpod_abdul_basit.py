#!/usr/bin/env python3
"""
WhisperX Abdul Basit Mujawwad Letter Timing Generator
Run on RunPod GPU - downloads audio from quranicaudio.com, processes with WhisperX

Audio URL: https://download.quranicaudio.com/quran/abdulbaset_mujawwad/{surah:03d}{ayah:03d}.mp3
Example: 001001.mp3 = Surah 1, Ayah 1
"""
import os, sys, json, subprocess
from pathlib import Path

# Setup
print("="*60)
print("ABDUL BASIT MUJAWWAD - WHISPERX PHONETIC LETTER TIMING")
print("="*60)

# Install whisperx and dependencies
print("\n[1/6] Installing WhisperX...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/m-bain/whisperx.git"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "phonemizer", "pydub"], check=True)

import whisperx
import torch
import urllib.request
from pydub import AudioSegment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[2/6] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Configuration
AUDIO_BASE_URL = "https://download.quranicaudio.com/quran/abdulbaset_mujawwad"
WORK_DIR = Path("/workspace/abdul_basit")
AUDIO_DIR = WORK_DIR / "audio"
OUTPUT_DIR = WORK_DIR / "output"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Surah verse counts (full 114 surahs)
SURAH_VERSES = {
    1: 7, 2: 286, 3: 200, 4: 176, 5: 120, 6: 165, 7: 206, 8: 75, 9: 129, 10: 109,
    11: 123, 12: 111, 13: 43, 14: 52, 15: 99, 16: 128, 17: 111, 18: 110, 19: 98, 20: 135,
    21: 112, 22: 78, 23: 118, 24: 64, 25: 77, 26: 227, 27: 93, 28: 88, 29: 69, 30: 60,
    31: 34, 32: 30, 33: 73, 34: 54, 35: 45, 36: 83, 37: 182, 38: 88, 39: 75, 40: 85,
    41: 54, 42: 53, 43: 89, 44: 59, 45: 37, 46: 35, 47: 38, 48: 29, 49: 18, 50: 45,
    51: 60, 52: 49, 53: 62, 54: 55, 55: 78, 56: 96, 57: 29, 58: 22, 59: 24, 60: 13,
    61: 14, 62: 11, 63: 11, 64: 18, 65: 12, 66: 12, 67: 30, 68: 52, 69: 52, 70: 44,
    71: 28, 72: 28, 73: 20, 74: 56, 75: 40, 76: 31, 77: 50, 78: 40, 79: 46, 80: 42,
    81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19, 88: 26, 89: 30, 90: 20,
    91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8, 99: 8, 100: 11,
    101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3, 109: 6, 110: 3,
    111: 5, 112: 4, 113: 5, 114: 6
}

# Phonetic weights for Arabic letters
PHONETIC_WEIGHTS = {
    'ا': 1.5, 'آ': 1.5, 'أ': 1.2, 'إ': 1.2, 'ء': 0.8,
    'ب': 1.0, 'ت': 1.0, 'ث': 1.0, 'ج': 1.0, 'ح': 1.0,
    'خ': 1.0, 'د': 1.0, 'ذ': 1.0, 'ر': 1.0, 'ز': 1.0,
    'س': 1.0, 'ش': 1.0, 'ص': 1.0, 'ض': 1.0, 'ط': 1.0,
    'ظ': 1.0, 'ع': 1.3, 'غ': 1.0, 'ف': 1.0, 'ق': 1.0,
    'ك': 1.0, 'ل': 1.0, 'م': 1.0, 'ن': 1.0, 'ه': 1.0,
    'و': 1.3, 'ي': 1.3, 'ى': 1.3, 'ة': 0.8, 'ئ': 0.9,
    'ُ': 0.3, 'َ': 0.3, 'ِ': 0.3, 'ً': 0.3, 'ٌ': 0.3, 'ٍ': 0.3,
    'ّ': 0.2, 'ْ': 0.2, 'ٰ': 0.5, 'ٓ': 0.3
}

def download_and_merge_surah(surah_num: int) -> Path:
    """Download all ayahs for a surah and merge into single MP3"""
    merged_path = AUDIO_DIR / f"surah_{surah_num:03d}.mp3"
    
    if merged_path.exists():
        print(f"  [SKIP] Already merged: {merged_path.name}")
        return merged_path
    
    num_ayahs = SURAH_VERSES.get(surah_num, 7)
    ayah_dir = AUDIO_DIR / f"surah_{surah_num:03d}_ayahs"
    ayah_dir.mkdir(exist_ok=True)
    
    # Download each ayah
    print(f"  [DOWNLOAD] Downloading {num_ayahs} ayahs...")
    for ayah in range(1, num_ayahs + 1):
        ayah_path = ayah_dir / f"{surah_num:03d}{ayah:03d}.mp3"
        if ayah_path.exists():
            continue
        # URL format: 001001.mp3 = surah 1, ayah 1
        url = f"{AUDIO_BASE_URL}/{surah_num:03d}{ayah:03d}.mp3"
        try:
            urllib.request.urlretrieve(url, ayah_path)
        except Exception as e:
            print(f"    Warning: Failed to download {surah_num}:{ayah} - {e}")
    
    # Merge all ayahs
    print(f"  [MERGE] Merging {num_ayahs} ayahs...")
    combined = AudioSegment.empty()
    for ayah in range(1, num_ayahs + 1):
        ayah_path = ayah_dir / f"{surah_num:03d}{ayah:03d}.mp3"
        if ayah_path.exists():
            segment = AudioSegment.from_mp3(ayah_path)
            combined += segment + AudioSegment.silent(duration=100)  # 100ms gap between ayahs
    
    combined.export(merged_path, format="mp3")
    print(f"  [MERGED] {merged_path.name} ({combined.duration_seconds:.1f}s)")
    return merged_path

def process_surah(surah_num: int, model, model_a, metadata):
    """Process a single surah and generate phonetic letter timing"""
    output_path = OUTPUT_DIR / f"letter_timing_{surah_num}.json"
    
    if output_path.exists():
        print(f"[SURAH {surah_num}] Already processed, skipping...")
        return True
    
    print(f"\n[SURAH {surah_num}] Processing...")
    
    try:
        # Download and merge audio
        audio_path = download_and_merge_surah(surah_num)
        
        # Process with WhisperX
        print(f"  [WHISPERX] Transcribing...")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16)
        
        print(f"  [ALIGN] Aligning characters...")
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=True)
        
        # Extract letter timing with phonetic weights
        letter_timing = []
        word_idx = 0
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                if "chars" in word:
                    for ci, ch in enumerate(word["chars"]):
                        char = ch.get("char", "")
                        weight = PHONETIC_WEIGHTS.get(char, 1.0)
                        letter_timing.append({
                            "charIdx": ci,
                            "char": char,
                            "start": int(ch.get("start", 0) * 1000),  # Convert to ms
                            "end": int(ch.get("end", 0) * 1000),
                            "duration": int((ch.get("end", 0) - ch.get("start", 0)) * 1000),
                            "weight": weight,
                            "wordIdx": word_idx
                        })
                word_idx += 1
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(letter_timing, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved {len(letter_timing)} letters to {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load models once
    print("\n[3/6] Loading Whisper large-v3 model...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type="float16", language="ar")
    print("[4/6] Loading Arabic alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    
    # Get surah range from args
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
        end = int(sys.argv[2]) if len(sys.argv) > 2 else start
        surahs = range(start, end + 1)
    else:
        # Default: start with short surahs for testing
        surahs = [1, 109, 110, 111, 112, 113, 114]
    
    print(f"\n[5/6] Processing surahs: {list(surahs)}...")
    
    results = {"success": [], "failed": []}
    for surah in surahs:
        if process_surah(surah, model, model_a, metadata):
            results["success"].append(surah)
        else:
            results["failed"].append(surah)
    
    # Summary
    print("\n" + "="*60)
    print("[6/6] COMPLETE!")
    print("="*60)
    print(f"Processed: {len(results['success'])}/{len(list(surahs))}")
    print(f"Output: {OUTPUT_DIR}")
    
    if results["failed"]:
        print(f"\nFailed: {results['failed']}")
    
    print("\nTo download output:")
    print("  runpodctl send /workspace/abdul_basit/output")

if __name__ == "__main__":
    main()
