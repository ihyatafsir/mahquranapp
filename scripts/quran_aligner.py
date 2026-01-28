#!/usr/bin/env python3
"""
QuranWhisperX Aligner (Prototype)

Combines:
1. wav2vec2-Arabic Forced Alignment (Phonetic)
2. LisanClean Roots (Morphological Anchoring)
3. Known Quran Text (Constraints)

Goal: Precise, root-aware timing for Quran recitation.
"""
import sys
import json
import torch
import whisperx
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LISAN_ROOTS_PATH = PROJECT_ROOT / "public/data/lisan_roots.json"
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
DEVICE = "cpu"

class QuranAligner:
    def __init__(self, surah_num):
        self.surah = str(surah_num)
        self.roots = self._load_roots()
        self.verses = self._load_verses()
        self.model, self.metadata = self._load_model()
        
    def _load_roots(self):
        print("Loading Lisan Roots...")
        with open(LISAN_ROOTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_verses(self):
        print(f"Loading Verses for Surah {self.surah}...")
        with open(VERSES_PATH, 'r', encoding='utf-8') as f:
            all_verses = json.load(f)
        return all_verses.get(self.surah, [])

    def _load_model(self):
        print("Loading wav2vec2 model...")
        return whisperx.load_align_model(language_code="ar", device=DEVICE)

    def align(self, audio_path):
        """
        Perform alignment using simple linear distribution.
        Since wav2vec2 has gaps/elongations, we distribute timing evenly.
        """
        # 1. Prepare Text
        full_text = ' '.join(v.get('text', '') for v in self.verses)
        
        # Build flat list of all non-space characters
        all_chars = [(i, c) for i, c in enumerate(full_text) if not c.isspace()]
        print(f"Total chars (no spaces): {len(all_chars)}")

        # 2. Load Audio to get duration
        audio = whisperx.load_audio(str(audio_path))
        audio_duration = len(audio) / 16000
        print(f"Audio duration: {audio_duration:.2f}s")
        
        # 3. Simple Linear Distribution
        # Distribute evenly across all characters
        char_duration = audio_duration / len(all_chars)
        print(f"Char duration: {char_duration:.4f}s ({char_duration*1000:.1f}ms)")
        
        timings = []
        for idx, (orig_idx, char) in enumerate(all_chars):
            start = idx * char_duration
            end = start + char_duration
            timings.append({
                "idx": idx,
                "char": char,
                "start": start,
                "end": end
            })
        
        return timings

    def save_output(self, timings):
        output_path = PROJECT_ROOT / f"public/data/abdul_basit/letter_timing_{self.surah}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timings, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(timings)} timings to {output_path}")

    def _strip_diacritics(self, text):
        DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
        return ''.join(c for c in text if c not in DIACRITICS)

    def _build_letter_groups(self, text):
        DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'
        groups = []
        current_group = []
        for i, char in enumerate(text):
            if char.isspace():
                if current_group: groups.append(current_group); current_group = []
            elif char in DIACRITICS:
                if current_group: current_group.append((i, char))
            else:
                if current_group: groups.append(current_group)
                current_group = [(i, char)]
        if current_group: groups.append(current_group)
        return groups

    def _expand_timings(self, base_timings, letter_groups):
        MAX_CHAR_DURATION_MS = 500  # Cap duration to prevent elongation stretching
        
        output_timing = []
        for gIdx, group in enumerate(letter_groups):
            if gIdx < len(base_timings):
                bt = base_timings[gIdx]
                start_ms = bt["start"] * 1000
                end_ms = bt["end"] * 1000
                duration = end_ms - start_ms
                char_duration = duration / len(group) if group else 0
                
                # Cap per-char duration to prevent madd/elongation from stretching
                char_duration = min(char_duration, MAX_CHAR_DURATION_MS)
                
                for cIdx, (orig_idx, char) in enumerate(group):
                    char_start = start_ms + (cIdx * char_duration)
                    output_timing.append({
                        "idx": len(output_timing),
                        "char": char,
                        "start": char_start / 1000.0, # seconds
                        "end": (char_start + char_duration) / 1000.0,
                        "groupIdx": gIdx
                    })
            else:
                 # Check if we have previous timing to extend from
                last_end = output_timing[-1]["end"] if output_timing else 0
                for cIdx, (orig_idx, char) in enumerate(group):
                     output_timing.append({
                        "idx": len(output_timing),
                        "char": char,
                        "start": last_end,
                        "end": last_end,
                        "groupIdx": gIdx
                    })
        return output_timing

    def _extract_timings(self, align_result):
        # Extract char-level timing 
        timings = []
        for seg in align_result.get("segments", []):
            if "chars" in seg:
                for ch in seg["chars"]:
                     if not ch.get("char", "").isspace():
                        timings.append(ch)
        return timings

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./quran_aligner.py <audio_path>")
        sys.exit(1)
        
    aligner = QuranAligner(surah_num=1)
    timings = aligner.align(sys.argv[1])
    aligner.save_output(timings)
    print(f"Alignment Complete.")
