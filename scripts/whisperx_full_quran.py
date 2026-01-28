#!/usr/bin/env python3
"""
WhisperX Full Quran Letter Timing Generator - RunPod Script
Processes: 
1. Abdul Basit Mujawwad (all 114 surahs from quranicaudio.com)
2. MAH audio files (uploaded from local)

Run on RunPod A40 GPU with 46GB VRAM
"""
import os, sys, json, subprocess, urllib.request
from pathlib import Path

def get_quran_surah_text(surah_num: int) -> list:
    """Fetch Quran text for a surah from alquran.cloud API
    Returns list of ayah texts for forced alignment"""
    url = f"https://api.alquran.cloud/v1/surah/{surah_num}/quran-uthmani"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            if data.get("code") == 200:
                ayahs = data.get("data", {}).get("ayahs", [])
                return [ayah.get("text", "") for ayah in ayahs]
    except Exception as e:
        print(f"  Warning: Failed to fetch Quran text: {e}")
    return []


print("="*70)
print("WHISPERX QURAN LETTER TIMING - FULL PROCESSING")
print("Abdul Basit Mujawwad + Mohammad Ahmad Hassan")
print("="*70)

# Dependencies should already be installed in the venv
# Do NOT run pip install here as it breaks version compatibility!
print("\n[SETUP] Loading pre-installed WhisperX...")

import torch
# Note: cloud_io.py is patched directly in the installed package with weights_only=False
print(f"torch version: {torch.__version__}")

import whisperx
from pydub import AudioSegment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Directories
WORKSPACE = Path("/workspace")
ABDUL_BASIT_DIR = WORKSPACE / "abdul_basit"
MAH_DIR = WORKSPACE / "mah"
OUTPUT_DIR = WORKSPACE / "output"
ABDUL_BASIT_DIR.mkdir(exist_ok=True)
MAH_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Surah verse counts
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
    'ا': 1.5, 'آ': 1.5, 'أ': 1.2, 'إ': 1.2, 'ء': 0.8, 'ٱ': 1.2,
    'ب': 1.0, 'ت': 1.0, 'ث': 1.0, 'ج': 1.0, 'ح': 1.0,
    'خ': 1.0, 'د': 1.0, 'ذ': 1.0, 'ر': 1.0, 'ز': 1.0,
    'س': 1.0, 'ش': 1.0, 'ص': 1.0, 'ض': 1.0, 'ط': 1.0,
    'ظ': 1.0, 'ع': 1.3, 'غ': 1.0, 'ف': 1.0, 'ق': 1.0,
    'ك': 1.0, 'ل': 1.0, 'م': 1.0, 'ن': 1.0, 'ه': 1.0,
    'و': 1.3, 'ي': 1.3, 'ى': 1.3, 'ة': 0.8, 'ئ': 0.9,
    'ُ': 0.3, 'َ': 0.3, 'ِ': 0.3, 'ً': 0.3, 'ٌ': 0.3, 'ٍ': 0.3,
    'ّ': 0.2, 'ْ': 0.2, 'ٰ': 0.5, 'ٓ': 0.3, 'ۖ': 0.1, 'ۗ': 0.1
}

def download_abdul_basit_surah(surah_num: int) -> Path:
    """Download full surah MP3 from quranicaudio.com/qdc/ (quran.com source)"""
    surah_path = ABDUL_BASIT_DIR / f"surah_{surah_num:03d}.mp3"
    if surah_path.exists() and surah_path.stat().st_size > 10000:  # At least 10KB
        print(f"  [SKIP] Already exists: {surah_path.name}")
        return surah_path
    
    # Correct URL: /qdc/ path (not /quran/) - verified from quran.com API
    url = f"https://download.quranicaudio.com/qdc/abdul_baset/mujawwad/{surah_num}.mp3"
    print(f"  [DOWNLOAD] {url}")
    
    try:
        urllib.request.urlretrieve(url, surah_path)
        file_size = surah_path.stat().st_size
        if file_size < 10000:  # Less than 10KB means error response
            print(f"  ✗ File too small ({file_size} bytes), likely 404 error")
            surah_path.unlink()  # Delete corrupted file
            return None
        print(f"  ✓ Downloaded {surah_path.name} ({file_size/(1024*1024):.1f} MB)")
        return surah_path
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None

from difflib import SequenceMatcher

def process_audio(audio_path: Path, output_name: str, reciter: str, surah_num: int, model, model_a, metadata):
    """Process audio with Standard Alignment + Sequence Matching.
    1. Transcribe & Align audio (unsupervised)
    2. Map resulting timing to Quran text using SequenceMatcher"""
    output_path = OUTPUT_DIR / reciter / f"letter_timing_{output_name}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    if output_path.exists():
        print(f"  [SKIP] Already processed: {output_path.name}")
        return True
    
    # Arabic diacritics
    DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
    
    def strip_diacritics(text):
        return ''.join(c for c in text if c not in DIACRITICS)
    
    def normalize_arabic(text):
        text = strip_diacritics(text)
        text = text.replace('آ', 'ا').replace('أ', 'ا').replace('إ', 'ا')
        return text

    def build_letter_groups(text):
        """Group each base letter with its following diacritics"""
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
    
    try:
        # Step 1: Get Quran text
        print(f"  [QURAN] Fetching surah {surah_num} text from API...")
        ayah_texts = get_quran_surah_text(surah_num)
        if not ayah_texts:
            print(f"  ✗ Failed to get Quran text for surah {surah_num}")
            return False
        
        full_quran_text = " ".join(ayah_texts)
        print(f"  [QURAN] text length: {len(full_quran_text)}")
        
        # Step 2: Load audio
        print(f"  [AUDIO] Loading {audio_path.name}...")
        audio = whisperx.load_audio(str(audio_path))
        
        # Step 3: Transcribe
        print(f"  [WHISPERX] Transcribing...")
        result = model.transcribe(audio, batch_size=16)
        segments = result.get("segments", [])
        
        # Step 4: Align (Standard)
        print(f"  [WHISPERX] Aligning...")
        aligned = whisperx.align(segments, model_a, metadata, audio, DEVICE, 
                                return_char_alignments=True)
        
        # Step 5: Extract Whisper Timing
        whisper_chars = [] # List of {char, start, end}
        for seg in aligned.get("segments", []):
            if "chars" in seg:
                for ch in seg["chars"]:
                    if "start" in ch and "end" in ch:
                        whisper_chars.append(ch)
        
        print(f"  [TIMING] Got {len(whisper_chars)} timed chars from Whisper")
        
        # Step 6: Sequence Match to Quran
        # Prepare for matching
        whisper_text = "".join(c["char"] for c in whisper_chars)
        whisper_norm = normalize_arabic(whisper_text)
        
        letter_groups = build_letter_groups(full_quran_text)
        quran_base_chars = [normalize_arabic(g[0][1]) for g in letter_groups]
        quran_norm = "".join(quran_base_chars)
        
        print(f"  [MATCH] Aligning {len(whisper_norm)} detected chars to {len(quran_norm)} Quran letters...")
        
        matcher = SequenceMatcher(None, whisper_norm, quran_norm, autojunk=False)
        blocks = matcher.get_matching_blocks()
        
        # Prepare output
        output_timing = []
        for i, char in enumerate(full_quran_text):
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
            # block.a is whisper index, block.b is quran group index
            for offset in range(block.size):
                w_idx = block.a + offset
                q_idx = block.b + offset
                
                if w_idx < len(whisper_chars) and q_idx < len(letter_groups):
                    # Get timing
                    w_char = whisper_chars[w_idx]
                    start_ms = int(w_char["start"] * 1000)
                    end_ms = int(w_char["end"] * 1000)
                    
                    # Distribute to group
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
                    
        print(f"  ✓ Matched {matched_count}/{len(letter_groups)} letters ({matched_count/len(letter_groups)*100:.1f}%)")
        
        # Interpolate gaps
        timed_indices = [i for i, t in enumerate(output_timing) if t["end"] > 0]
        if len(timed_indices) > 2:
            for i in range(len(output_timing)):
                if output_timing[i]["is_space"] or output_timing[i]["end"] > 0: continue
                
                # Find neighbors
                prev_end = 0
                next_start = 0
                prev_idx = -1
                next_idx = -1
                
                # Back
                for p in range(i-1, -1, -1):
                    if output_timing[p]["end"] > 0:
                        prev_end = output_timing[p]["end"]
                        prev_idx = p
                        break
                # Fwd
                for n in range(i+1, len(output_timing)):
                    if output_timing[n]["end"] > 0:
                        next_start = output_timing[n]["start"]
                        next_idx = n
                        break
                        
                if prev_idx != -1 and next_idx != -1:
                    gap_len = sum(1 for k in range(prev_idx+1, next_idx) if not output_timing[k]["is_space"])
                    if gap_len > 0:
                        pos = sum(1 for k in range(prev_idx+1, i+1) if not output_timing[k]["is_space"])
                        duration = max(0, next_start - prev_end)
                        step = duration / (gap_len + 1)
                        
                        est_start = prev_end + step * (pos - 0.5)
                        est_end = prev_end + step * (pos + 0.5)
                        output_timing[i]["start"] = int(est_start)
                        output_timing[i]["end"] = int(est_end)
                        output_timing[i]["duration"] = int(step)

        # Filter out spaces for final output
        final_output = [t for t in output_timing if not t["is_space"]]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved {len(final_output)} letters -> {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load models
    print("\n[MODELS] Loading Whisper large-v3...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type="float16", language="ar")
    print("[MODELS] Loading Arabic alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    
    # Get mode from args
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 114
    
    results = {"abdul_basit": {"success": [], "failed": []}, 
               "mah": {"success": [], "failed": []}}
    
    # Process Abdul Basit
    if mode in ["all", "abdul_basit", "ab"]:
        print(f"\n{'='*70}")
        print(f"ABDUL BASIT MUJAWWAD - Surahs {start}-{end}")
        print("="*70)
        
        for surah in range(start, end + 1):
            print(f"\n[SURAH {surah}] Abdul Basit Mujawwad")
            audio_path = download_abdul_basit_surah(surah)
            if process_audio(audio_path, str(surah), "abdul_basit", surah, model, model_a, metadata):
                results["abdul_basit"]["success"].append(surah)
            else:
                results["abdul_basit"]["failed"].append(surah)
    
    # Process MAH files
    if mode in ["all", "mah"]:
        print(f"\n{'='*70}")
        print("MOHAMMAD AHMAD HASSAN - Processing uploaded files")
        print("="*70)
        
        mah_files = list(MAH_DIR.glob("*.mp3")) + list(WORKSPACE.glob("surah_*.mp3"))
        for audio_path in sorted(mah_files):
            # Extract surah number from filename
            name = audio_path.stem
            if "surah_" in name or "surah-" in name:
                parts = name.replace("surah_", "").replace("surah-", "").split("_")[0].split("-")[0]
                surah_num = ''.join(filter(str.isdigit, parts[:3]))
            else:
                surah_num = name
            
            print(f"\n[MAH] Processing {audio_path.name}")
            if process_audio(audio_path, surah_num, "mah", int(surah_num) if surah_num.isdigit() else 2, model, model_a, metadata):
                results["mah"]["success"].append(surah_num)
            else:
                results["mah"]["failed"].append(surah_num)
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Abdul Basit: {len(results['abdul_basit']['success'])} succeeded, {len(results['abdul_basit']['failed'])} failed")
    print(f"MAH: {len(results['mah']['success'])} succeeded, {len(results['mah']['failed'])} failed")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("\nTo download: runpodctl receive /workspace/output")

if __name__ == "__main__":
    main()
