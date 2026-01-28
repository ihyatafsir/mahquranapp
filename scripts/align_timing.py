#!/usr/bin/env python3
"""
Millisecond-Accurate Quran Letter Alignment

Strategy:
1. WhisperX gives timing for base letters (consonants)
2. Quran text has base letters + diacritics (tashkeel)
3. We match base letters and distribute timing to cover diacritics

Example:
  WhisperX: "ب" at 500-600ms
  Quran: "بِ" (ba + kasra)
  Result: "ب" at 500-550ms, "ِ" at 550-600ms
"""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

# Paths
DATA_DIR = Path(__file__).parent.parent / "public" / "data"
ABDUL_BASIT_ORIG_DIR = DATA_DIR / "abdul_basit_original"
ABDUL_BASIT_DIR = DATA_DIR / "abdul_basit"
VERSES_FILE = DATA_DIR / "verses_v4.json"

# Arabic diacritics (tashkeel) - these don't have their own timing
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670')

# Additional marks that should share timing with previous letter
COMBINING_MARKS = DIACRITICS | set('\u0640')  # tatweel

def is_base_letter(char):
    """Check if character is a base letter (not a diacritic or space)"""
    return char not in DIACRITICS and not char.isspace() and char != '\u0640'

def strip_diacritics(text):
    """Remove diacritics for matching"""
    return ''.join(c for c in text if c not in DIACRITICS)

def normalize_for_matching(text):
    """Normalize text for matching: remove diacritics, spaces, normalize alef"""
    text = strip_diacritics(text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ؤئء]', 'ء', text)
    return text

def build_base_letter_indices(text):
    """
    Build mapping: base_letter_index -> list of (char_index, char) for that letter group
    Each group = base letter + following diacritics
    """
    groups = []
    current_group = []
    
    for i, char in enumerate(text):
        if char.isspace():
            # Space - start new group
            if current_group:
                groups.append(current_group)
                current_group = []
        elif is_base_letter(char):
            # Base letter - start new group
            if current_group:
                groups.append(current_group)
            current_group = [(i, char)]
        else:
            # Diacritic - add to current group
            if current_group:
                current_group.append((i, char))
    
    if current_group:
        groups.append(current_group)
    
    return groups

def align_timing_to_quran(surah_num, timing_data, quran_text):
    """
    Align WhisperX timing to Quran text with millisecond accuracy.
    
    Returns list of timing entries for EVERY character in quran_text (including spaces).
    """
    # Build letter groups from Quran text
    letter_groups = build_base_letter_indices(quran_text)
    
    # Get WhisperX timing entries (filter out empty)
    whisper_entries = [t for t in timing_data if t.get('char', '').strip()]
    
    # Normalize both for matching
    whisper_base_letters = [normalize_for_matching(t['char']) for t in whisper_entries]
    quran_base_letters = [normalize_for_matching(g[0][1]) for g in letter_groups]
    
    whisper_str = ''.join(whisper_base_letters)
    quran_str = ''.join(quran_base_letters)
    
    print(f"  WhisperX: {len(whisper_entries)} timed letters")
    print(f"  Quran: {len(letter_groups)} letter groups, {len(quran_text)} total chars")
    
    # Use SequenceMatcher to find aligned blocks
    matcher = SequenceMatcher(None, whisper_str, quran_str, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    
    # Initialize output with empty timing values
    output = []
    for i, char in enumerate(quran_text):
        output.append({
            'idx': i,
            'char': char,
            'start': 0,
            'end': 0,
            'duration': 0,
            'is_space': char.isspace()
        })
    
    # Map matched Quran letter groups to WhisperX timing
    matched_groups = 0
    total_groups = len(letter_groups)
    
    for block in matching_blocks:
        whisper_start, quran_start, size = block.a, block.b, block.size
        
        for offset in range(size):
            whisper_idx = whisper_start + offset
            quran_group_idx = quran_start + offset
            
            if whisper_idx >= len(whisper_entries) or quran_group_idx >= len(letter_groups):
                continue
            
            # Get timing from WhisperX (already in ms)
            timing = whisper_entries[whisper_idx]
            start_ms = timing.get('start', 0) or 0
            end_ms = timing.get('end', 0) or 0
            
            # Ensure we're working in milliseconds
            if start_ms < 1000 and start_ms > 0:  # Likely seconds, convert to ms
                start_ms = start_ms * 1000
                end_ms = end_ms * 1000
            
            # Get the character group (base letter + diacritics)
            group = letter_groups[quran_group_idx]
            group_size = len(group)
            
            # Distribute timing across the group (in milliseconds)
            if group_size > 0 and end_ms > start_ms:
                duration_per_char = (end_ms - start_ms) / group_size
                
                for j, (char_idx, char) in enumerate(group):
                    char_start = start_ms + (j * duration_per_char)
                    char_end = char_start + duration_per_char
                    
                    output[char_idx]['start'] = int(char_start)
                    output[char_idx]['end'] = int(char_end)
                    output[char_idx]['duration'] = int(duration_per_char)
            
            matched_groups += 1
    
    match_pct = (matched_groups / total_groups * 100) if total_groups > 0 else 0
    print(f"  ✓ Matched {matched_groups}/{total_groups} letter groups ({match_pct:.1f}%)")
    
    # Interpolate gaps for unmatched groups
    interpolate_gaps(output)
    
    # Filter out spaces for final output (App expects non-space chars only)
    final_output = [t for t in output if not t['is_space']]
    
    return final_output

def interpolate_gaps(timing_list):
    """Fill in timing gaps by linear interpolation"""
    # Find indices with valid timing
    timed_indices = [i for i, t in enumerate(timing_list) if t['end'] > 0]
    
    if len(timed_indices) < 2:
        return
    
    for i, entry in enumerate(timing_list):
        if entry['is_space'] or entry['end'] > 0:
            continue
        
        # Find previous and next timed entries
        prev_idx = None
        next_idx = None
        
        for j in range(i - 1, -1, -1):
            if timing_list[j]['end'] > 0:
                prev_idx = j
                break
        
        for j in range(i + 1, len(timing_list)):
            if timing_list[j]['end'] > 0:
                next_idx = j
                break
        
        if prev_idx is not None and next_idx is not None:
            # Interpolate
            prev_end = timing_list[prev_idx]['end']
            next_start = timing_list[next_idx]['start']
            
            # Count gap characters
            gap_chars = sum(1 for k in range(prev_idx + 1, next_idx) if not timing_list[k]['is_space'])
            if gap_chars > 0:
                # Find position in gap
                pos = sum(1 for k in range(prev_idx + 1, i + 1) if not timing_list[k]['is_space'])
                
                gap_duration = max(0, next_start - prev_end)
                step = gap_duration / (gap_chars + 1)
                
                entry['start'] = int(prev_end + step * (pos - 0.5))
                entry['end'] = int(prev_end + step * (pos + 0.5))
                entry['duration'] = int(step)

def process_surah(surah_num, verses, timing_path, output_path):
    """Process one surah"""
    with open(timing_path) as f:
        timing_data = json.load(f)
    
    # Join verses with space (standard)
    full_text = ' '.join(v['text'] for v in verses)
    
    print(f"Surah {surah_num}: {len(timing_data)} timing entries")
    
    aligned = align_timing_to_quran(surah_num, timing_data, full_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(aligned)} entries -> {output_path.name}")

def main():
    print("=" * 60)
    print("MILLISECOND-ACCURATE QURAN LETTER ALIGNMENT")
    print("=" * 60)
    
    with open(VERSES_FILE) as f:
        all_verses = json.load(f)
    
    for timing_file in sorted(ABDUL_BASIT_ORIG_DIR.glob("letter_timing_*.json")):
        surah_num = int(timing_file.stem.split('_')[-1])
        
        verses = all_verses.get(str(surah_num), [])
        if not verses:
            print(f"Skipping surah {surah_num} - no verse data")
            continue
        
        output_path = ABDUL_BASIT_DIR / f"letter_timing_{surah_num}.json"
        process_surah(surah_num, verses, timing_file, output_path)
    
    print("\n" + "=" * 60)
    print("ALIGNMENT COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
