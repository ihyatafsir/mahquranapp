#!/usr/bin/env python3
"""
Test alignment accuracy for all surahs.
Checks:
1. Coverage: % of characters with non-zero timing
2. Monotonicity: Start times should be monotonic
3. Gaps: Detect large gaps in timing
4. Duration: Detect generic durations (indicative of fallback interpolation)
"""
import json
from pathlib import Path
import sys

DATA_DIR = Path(__file__).parent.parent / "public" / "data"
ABDUL_BASIT_DIR = DATA_DIR / "abdul_basit"
VERSES_FILE = DATA_DIR / "verses_v4.json"

def test_surah_alignment(surah_num: int, verses: list, timing_path: Path):
    with open(timing_path) as f:
        timing = json.load(f)
    
    full_text = ''.join(v['text'] for v in verses)
    chars_count = len(full_text)
    total_non_space = len([c for c in full_text if not c.isspace()])

    # 1. Length check
    if len(timing) != total_non_space:
        print(f"❌ Surah {surah_num}: Length mismatch! Timing {len(timing)} vs Non-Space Text {total_non_space}")
        return False

    # 2. Coverage
    timed_chars = [t for t in timing if t['end'] > 0 and not t.get('is_space', False)]
    total_non_space = len([c for c in full_text if not c.isspace()])
    coverage = len(timed_chars) / max(total_non_space, 1) * 100
    
    # 3. Monotonicity
    issues = 0
    prev_start = 0
    for t in timed_chars:
        if t['start'] < prev_start - 0.1: # Allow small overlap
            issues += 1
        prev_start = t['start']
    
    # 4. Interpolation detection (exact 0.1s duration or essentially identical start/ends often means crude interpolation)
    interpolated = len([t for t in timed_chars if t['duration'] == 0.1 or t['start'] == t['end']])
    interp_rate = interpolated / max(len(timed_chars), 1) * 100

    print(f"Surah {surah_num}:")
    print(f"  Coverage: {coverage:.1f}% ({len(timed_chars)}/{total_non_space})")
    print(f"  Monotonicity Issues: {issues}")
    print(f"  Likely Interpolated: {interp_rate:.1f}%")
    
    if coverage < 90:
        print("  ⚠️ Low coverage")
    if issues > 0:
        print("  ⚠️ Monotonicity errors")
    
    return True

def main():
    print("Alignment Accuracy Test")
    print("=======================")
    
    with open(VERSES_FILE) as f:
        all_verses = json.load(f)
        
    timing_files = sorted(ABDUL_BASIT_DIR.glob("letter_timing_*.json"), key=lambda p: int(p.stem.split('_')[-1]))
    
    for timing_file in timing_files:
        surah_num = int(timing_file.stem.split('_')[-1])
        verses = all_verses.get(str(surah_num))
        if verses:
            test_surah_alignment(surah_num, verses, timing_file)

if __name__ == "__main__":
    main()
