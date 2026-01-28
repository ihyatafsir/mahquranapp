#!/usr/bin/env python3
"""
Comprehensive Quran Text vs Timing Data Alignment Test Tool

This tool analyzes the alignment between:
1. Quran text from verses_v4.json
2. Letter timing data from letter_timing_X.json

It helps diagnose why highlighting might be off.
"""

import json
import sys
import os
from pathlib import Path

# Arabic diacritics (tashkeel) - same as in App.tsx
DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def is_diacritic(char):
    return char in DIACRITICS

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_surah_text(verses_data, surah_num):
    """Extract full text from surah, preserving structure."""
    surah_data = verses_data.get(str(surah_num), [])
    result = {
        'verses': [],
        'full_text': '',
        'letters_only': '',  # Non-space characters only
        'letter_positions': []  # (verse_idx, char_idx_in_verse, global_idx)
    }
    
    global_idx = 0
    for verse_idx, verse in enumerate(surah_data):
        verse_text = verse.get('text', '')
        result['verses'].append({
            'ayah': verse.get('ayah'),
            'text': verse_text,
            'letter_count': len(verse_text.replace(' ', '').replace('\n', ''))
        })
        result['full_text'] += verse_text
        
        for char_idx, char in enumerate(verse_text):
            if not char.isspace():
                result['letters_only'] += char
                result['letter_positions'].append({
                    'verse_idx': verse_idx,
                    'verse_char_idx': char_idx,
                    'global_letter_idx': global_idx,
                    'char': char,
                    'is_diacritic': is_diacritic(char)
                })
                global_idx += 1
    
    return result

def analyze_timing_data(timing_data):
    """Analyze timing data structure."""
    result = {
        'total_entries': len(timing_data),
        'letters': '',
        'entries': [],
        'words': {},
        'has_charIdx': False,
        'has_wordIdx': False,
        'time_range': {'start': None, 'end': None}
    }
    
    if not timing_data:
        return result
    
    # Check structure
    first = timing_data[0]
    result['has_charIdx'] = 'charIdx' in first
    result['has_wordIdx'] = 'wordIdx' in first
    
    for i, entry in enumerate(timing_data):
        char = entry.get('char', '')
        result['letters'] += char
        result['entries'].append({
            'idx': i,
            'char': char,
            'wordIdx': entry.get('wordIdx', -1),
            'charIdx': entry.get('charIdx', -1),
            'start': entry.get('start', 0),
            'end': entry.get('end', 0)
        })
        
        # Track words
        word_idx = entry.get('wordIdx', 0)
        if word_idx not in result['words']:
            result['words'][word_idx] = {'letters': '', 'start': entry.get('start'), 'end': entry.get('end')}
        result['words'][word_idx]['letters'] += char
        result['words'][word_idx]['end'] = entry.get('end')
    
    if timing_data:
        result['time_range']['start'] = timing_data[0].get('start', 0)
        result['time_range']['end'] = timing_data[-1].get('end', 0)
    
    return result

def compare_alignment(quran_data, timing_analysis):
    """Compare Quran text with timing data letter by letter."""
    quran_letters = quran_data['letters_only']
    timing_letters = timing_analysis['letters']
    
    result = {
        'quran_letter_count': len(quran_letters),
        'timing_letter_count': len(timing_letters),
        'count_match': len(quran_letters) == len(timing_letters),
        'mismatches': [],
        'first_mismatch_idx': -1,
        'alignment_summary': ''
    }
    
    # Find mismatches
    min_len = min(len(quran_letters), len(timing_letters))
    for i in range(min_len):
        q_char = quran_letters[i]
        t_char = timing_letters[i]
        if q_char != t_char:
            result['mismatches'].append({
                'idx': i,
                'quran_char': q_char,
                'quran_unicode': hex(ord(q_char)),
                'timing_char': t_char,
                'timing_unicode': hex(ord(t_char))
            })
            if result['first_mismatch_idx'] == -1:
                result['first_mismatch_idx'] = i
    
    # Summary
    if result['count_match'] and len(result['mismatches']) == 0:
        result['alignment_summary'] = '✅ PERFECT ALIGNMENT'
    elif result['count_match']:
        result['alignment_summary'] = f'⚠️ Same count but {len(result["mismatches"])} character mismatches'
    else:
        diff = abs(len(quran_letters) - len(timing_letters))
        if len(quran_letters) > len(timing_letters):
            result['alignment_summary'] = f'❌ Timing data MISSING {diff} characters'
        else:
            result['alignment_summary'] = f'❌ Timing data has {diff} EXTRA characters'
    
    return result

def print_detailed_comparison(quran_data, timing_analysis, comparison, show_all=False):
    """Print detailed comparison output."""
    print("=" * 80)
    print("QURAN TEXT vs TIMING DATA ALIGNMENT ANALYSIS")
    print("=" * 80)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Quran letters (non-space): {comparison['quran_letter_count']}")
    print(f"   Timing entries:            {comparison['timing_letter_count']}")
    print(f"   Match: {'✅ YES' if comparison['count_match'] else '❌ NO'}")
    
    print(f"\n📈 TIMING RANGE:")
    print(f"   Start: {timing_analysis['time_range']['start']:.3f}s")
    print(f"   End:   {timing_analysis['time_range']['end']:.3f}s")
    
    print(f"\n📋 ALIGNMENT SUMMARY: {comparison['alignment_summary']}")
    
    if comparison['first_mismatch_idx'] >= 0:
        print(f"\n⚠️ FIRST MISMATCH at index {comparison['first_mismatch_idx']}:")
        mm = comparison['mismatches'][0]
        print(f"   Quran:  '{mm['quran_char']}' ({mm['quran_unicode']})")
        print(f"   Timing: '{mm['timing_char']}' ({mm['timing_unicode']})")
        
        # Context
        idx = comparison['first_mismatch_idx']
        q = quran_data['letters_only']
        t = timing_analysis['letters']
        start = max(0, idx - 5)
        end = min(len(q), len(t), idx + 10)
        print(f"\n   Context around mismatch:")
        print(f"   Quran:  ...{q[start:end]}...")
        print(f"   Timing: ...{t[start:end]}...")
    
    # Show word breakdown
    print(f"\n📝 TIMING WORDS (first 10):")
    for word_idx in sorted(timing_analysis['words'].keys())[:10]:
        word = timing_analysis['words'][word_idx]
        print(f"   Word {word_idx}: '{word['letters']}' ({word['start']:.2f}s - {word['end']:.2f}s)")
    
    # Show letter-by-letter comparison (first 50)
    if show_all or len(comparison['mismatches']) > 0:
        print(f"\n🔍 LETTER-BY-LETTER (first 50):")
        print(f"{'Idx':<5} | {'Quran':<6} | {'Timing':<6} | {'Match'}")
        print("-" * 35)
        for i in range(min(50, comparison['quran_letter_count'], comparison['timing_letter_count'])):
            q = quran_data['letters_only'][i]
            t = timing_analysis['letters'][i]
            match = '✅' if q == t else '❌'
            diag = '◌' if is_diacritic(q) else ' '
            print(f"{i:<5} | {q}{diag:<5} | {t:<6} | {match}")

def suggest_fix(comparison, quran_data, timing_analysis):
    """Suggest how to fix the alignment."""
    print("\n" + "=" * 80)
    print("💡 SUGGESTED FIXES")
    print("=" * 80)
    
    if comparison['alignment_summary'].startswith('✅'):
        print("No fixes needed - alignment is perfect!")
        return
    
    # Check if timing starts later (Isti'aatha issue)
    t_start = timing_analysis['time_range']['start']
    if t_start > 3.0:
        print("\n1️⃣ ISSUE: Timing data starts at {:.2f}s (likely skips Isti'aatha)".format(t_start))
        print("   This MAH audio includes Isti'aatha but timing only covers Quran text.")
        print("   FIX: Timing data should only align with Quran text, not Isti'aatha.")
        print("   The current timing appears to be CORRECT for text alignment.")
    
    if comparison['first_mismatch_idx'] >= 0:
        mm = comparison['mismatches'][0]
        q_u = ord(mm['quran_char'])
        t_u = ord(mm['timing_char'])
        
        # Common issues
        if q_u == 0x0627 and t_u == 0x0671:  # Alef vs Alef Wasla
            print("\n2️⃣ ISSUE: Unicode normalization mismatch")
            print("   Quran uses regular Alef (\\u0627)")
            print("   Timing uses Alef Wasla (\\u0671)")
            print("   FIX: Normalize one set to match the other")
        
        if is_diacritic(mm['quran_char']) or is_diacritic(mm['timing_char']):
            print("\n3️⃣ ISSUE: Diacritic handling difference")
            print("   One source includes diacritics that the other doesn't")
            print("   FIX: Ensure both sources handle diacritics the same way")
    
    # Count diff suggests offset
    diff = comparison['quran_letter_count'] - comparison['timing_letter_count']
    if diff > 0:
        print(f"\n4️⃣ ISSUE: Quran has {diff} more characters than timing")
        print("   This might mean timing is missing some letters/diacritics")
    elif diff < 0:
        print(f"\n4️⃣ ISSUE: Timing has {-diff} extra characters")
        print("   Timing might include Isti'aatha or extra content not in Quran text")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Quran text vs timing alignment')
    parser.add_argument('surah', type=int, nargs='?', default=1, help='Surah number (default: 1)')
    parser.add_argument('--all', action='store_true', help='Show all letter comparisons')
    parser.add_argument('--reciter', default='mah', choices=['mah', 'abdul_basit'], help='Reciter')
    args = parser.parse_args()
    
    # Paths
    base_path = Path(__file__).parent.parent / 'public' / 'data'
    verses_path = base_path / 'verses_v4.json'
    
    if args.reciter == 'abdul_basit':
        timing_path = base_path / 'abdul_basit' / f'letter_timing_{args.surah}.json'
    else:
        timing_path = base_path / f'letter_timing_{args.surah}.json'
    
    print(f"Loading Surah {args.surah} ({args.reciter})...")
    print(f"  Verses: {verses_path}")
    print(f"  Timing: {timing_path}")
    
    if not verses_path.exists():
        print(f"ERROR: Verses file not found: {verses_path}")
        return 1
    
    if not timing_path.exists():
        print(f"ERROR: Timing file not found: {timing_path}")
        return 1
    
    # Load data
    verses = load_json(verses_path)
    timing = load_json(timing_path)
    
    # Analyze
    quran_data = extract_surah_text(verses, args.surah)
    timing_analysis = analyze_timing_data(timing)
    comparison = compare_alignment(quran_data, timing_analysis)
    
    # Print results
    print_detailed_comparison(quran_data, timing_analysis, comparison, args.all)
    suggest_fix(comparison, quran_data, timing_analysis)
    
    return 0 if comparison['alignment_summary'].startswith('✅') else 1

if __name__ == '__main__':
    sys.exit(main())
