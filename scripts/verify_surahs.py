#!/usr/bin/env python3
"""
Verify all surahs have matching data:
1. Audio file exists
2. Timing file exists and has correct format
3. Verses data exists
4. Word count matches between timing and verses
"""

import json
import os
from pathlib import Path

DATA_DIR = Path('/home/absolut7/Documents/26apps/MahQuranApp/public/data')
AUDIO_DIR = Path('/home/absolut7/Documents/26apps/MahQuranApp/public/audio')

SURAHS = [1, 18, 36, 47, 53, 55, 56, 67, 71, 75, 80, 82, 85, 87, 89, 90, 91, 92, 93, 109, 112, 113, 114]

def check_surah(surah_num: int) -> dict:
    """Check all data for a surah."""
    result = {
        'surah': surah_num,
        'audio': False,
        'timing': False,
        'verses': False,
        'timing_words': 0,
        'verse_words': 0,
        'match': False,
        'errors': []
    }
    
    # Check audio
    audio_path = AUDIO_DIR / f'surah_{surah_num:03d}.mp3'
    result['audio'] = audio_path.exists()
    if not result['audio']:
        result['errors'].append(f'Missing audio: {audio_path.name}')
    
    # Check timing
    timing_path = DATA_DIR / f'letter_timing_{surah_num}.json'
    if timing_path.exists():
        result['timing'] = True
        try:
            with open(timing_path) as f:
                timing = json.load(f)
            # Count unique wordIdx
            word_indices = set(t['wordIdx'] for t in timing)
            result['timing_words'] = len(word_indices)
            
            # Check timing format
            if len(timing) > 0:
                first = timing[0]
                if 'char' not in first or 'start' not in first:
                    result['errors'].append('Invalid timing format')
        except Exception as e:
            result['errors'].append(f'Timing error: {e}')
    else:
        result['errors'].append(f'Missing timing: {timing_path.name}')
    
    # Check verses
    try:
        with open(DATA_DIR / 'verses_v4.json') as f:
            all_verses = json.load(f)
        verses = all_verses.get(str(surah_num), [])
        if verses:
            result['verses'] = True
            # Count total words
            total_words = sum(len(v['text'].split()) for v in verses)
            result['verse_words'] = total_words
        else:
            result['errors'].append('No verses found')
    except Exception as e:
        result['errors'].append(f'Verses error: {e}')
    
    # Check word count match
    if result['timing_words'] > 0 and result['verse_words'] > 0:
        diff = abs(result['timing_words'] - result['verse_words'])
        # Allow small difference (1-2 words)
        result['match'] = diff <= 2
        if not result['match']:
            result['errors'].append(f'Word count mismatch: timing={result["timing_words"]}, verses={result["verse_words"]} (diff={diff})')
    
    return result

def main():
    print("=" * 70)
    print("MAH Quran App - Surah Data Verification")
    print("=" * 70)
    
    all_ok = True
    results = []
    
    for surah in SURAHS:
        result = check_surah(surah)
        results.append(result)
        
        status = "✓" if not result['errors'] else "✗"
        audio_status = "✓" if result['audio'] else "✗"
        timing_status = "✓" if result['timing'] else "✗"
        verses_status = "✓" if result['verses'] else "✗"
        match_status = "✓" if result['match'] else "✗"
        
        print(f"\nSurah {surah:3d}: {status}")
        print(f"  Audio: {audio_status} | Timing: {timing_status} | Verses: {verses_status} | Match: {match_status}")
        print(f"  Words: timing={result['timing_words']}, verses={result['verse_words']}")
        
        if result['errors']:
            all_ok = False
            for err in result['errors']:
                print(f"  ⚠ {err}")
    
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if not r['errors'])
    print(f"Results: {passed}/{len(SURAHS)} surahs passed all checks")
    
    if all_ok:
        print("✓ All surahs verified successfully!")
    else:
        print("✗ Some issues found - see above for details")
    
    print("=" * 70)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    exit(main())
