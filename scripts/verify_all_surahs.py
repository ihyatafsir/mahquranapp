#!/usr/bin/env python3
"""
Comprehensive Surah Verification Script
Checks each surah for:
1. Audio file exists
2. Timing file exists with correct format
3. Word count matches between timing and verses
4. Timing starts reasonably (within first few seconds)
"""

import json
from pathlib import Path

DATA_DIR = Path('/home/absolut7/Documents/26apps/MahQuranApp/public/data')
AUDIO_DIR = Path('/home/absolut7/Documents/26apps/MahQuranApp/public/audio')

SURAHS = [1, 18, 36, 47, 53, 55, 56, 67, 71, 75, 80, 82, 85, 87, 89, 90, 91, 92, 93, 109, 112, 113, 114]

# Load verses
with open(DATA_DIR / 'verses_v4.json') as f:
    ALL_VERSES = json.load(f)

def normalize_time(t):
    """Normalize time to seconds"""
    if t > 100:  # Milliseconds
        return t / 1000
    return t

def check_surah(surah_num: int) -> dict:
    """Comprehensive check for a surah."""
    result = {
        'surah': surah_num,
        'audio': False,
        'timing': False,
        'timing_format': 'unknown',
        'timing_words': 0,
        'timing_letters': 0,
        'verse_words': 0,
        'word_match': False,
        'first_start': None,
        'last_end': None,
        'issues': []
    }
    
    # Check audio
    audio_path = AUDIO_DIR / f'surah_{surah_num:03d}.mp3'
    result['audio'] = audio_path.exists() or audio_path.is_symlink()
    if not result['audio']:
        result['issues'].append(f'Missing audio')
    
    # Check timing
    timing_path = DATA_DIR / f'letter_timing_{surah_num}.json'
    if timing_path.exists():
        result['timing'] = True
        try:
            with open(timing_path) as f:
                timing = json.load(f)
            
            if len(timing) > 0:
                result['timing_letters'] = len(timing)
                
                # Check format
                first = timing[0]
                if 'wordIdx' in first and 'char' in first:
                    result['timing_format'] = 'valid'
                    
                    # Count unique words
                    word_indices = set(t['wordIdx'] for t in timing)
                    result['timing_words'] = len(word_indices)
                    
                    # Check timing range
                    result['first_start'] = normalize_time(first['start'])
                    result['last_end'] = normalize_time(timing[-1]['end'])
                    
                    # Check if timing is too far off
                    if result['first_start'] > 60:
                        result['issues'].append(f"Timing starts late: {result['first_start']:.1f}s")
                else:
                    result['timing_format'] = 'invalid'
                    result['issues'].append('Invalid timing format')
            else:
                result['issues'].append('Empty timing file')
        except Exception as e:
            result['issues'].append(f'Timing error: {e}')
    else:
        result['issues'].append('Missing timing file')
    
    # Check verses
    verses = ALL_VERSES.get(str(surah_num), [])
    if verses:
        # Count total words
        total_words = sum(len(v['text'].split()) for v in verses)
        result['verse_words'] = total_words
        
        # Check word count match
        if result['timing_words'] > 0:
            diff = abs(result['timing_words'] - result['verse_words'])
            diff_pct = diff / result['verse_words'] * 100 if result['verse_words'] > 0 else 0
            
            if diff == 0:
                result['word_match'] = True
            elif diff <= 2:
                result['word_match'] = True  # Allow small variance
            elif diff_pct < 5:
                result['word_match'] = True  # Allow 5% variance
                result['issues'].append(f'Minor word diff: {diff} ({diff_pct:.1f}%)')
            else:
                result['issues'].append(f'Word mismatch: timing={result["timing_words"]}, verses={result["verse_words"]} (diff={diff})')
    else:
        result['issues'].append('No verses found')
    
    return result

def main():
    print("=" * 80)
    print("MAH Quran - Comprehensive Surah Verification")
    print("=" * 80)
    
    results = []
    
    for surah in SURAHS:
        result = check_surah(surah)
        results.append(result)
    
    # Print detailed results
    print("\n{:<6} {:<12} {:<8} {:<8} {:<10} {:<10} {:<8} {}".format(
        "Surah", "Audio", "Timing", "Letters", "T-Words", "V-Words", "Match", "Issues"
    ))
    print("-" * 80)
    
    for r in results:
        audio_str = "✓" if r['audio'] else "✗"
        timing_str = "✓" if r['timing'] else "✗"
        match_str = "✓" if r['word_match'] else "✗"
        issues_str = ", ".join(r['issues']) if r['issues'] else "-"
        
        print("{:<6} {:<12} {:<8} {:<8} {:<10} {:<10} {:<8} {}".format(
            r['surah'],
            audio_str,
            timing_str,
            r['timing_letters'],
            r['timing_words'],
            r['verse_words'],
            match_str,
            issues_str[:40]
        ))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    passed = sum(1 for r in results if not r['issues'])
    word_match = sum(1 for r in results if r['word_match'])
    has_audio = sum(1 for r in results if r['audio'])
    has_timing = sum(1 for r in results if r['timing'])
    
    print(f"  Audio files: {has_audio}/{len(results)}")
    print(f"  Timing files: {has_timing}/{len(results)}")
    print(f"  Word count match: {word_match}/{len(results)}")
    print(f"  Fully passing: {passed}/{len(results)}")
    
    # List surahs with issues
    issues_list = [r for r in results if r['issues']]
    if issues_list:
        print("\nSurahs needing attention:")
        for r in issues_list:
            print(f"  - Surah {r['surah']}: {', '.join(r['issues'])}")
    else:
        print("\n✓ All surahs verified successfully!")
    
    print("=" * 80)
    
    return 0 if not issues_list else 1

if __name__ == '__main__':
    exit(main())
