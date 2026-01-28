#!/usr/bin/env python3
"""
Comprehensive diagnostic for Surah 1 highlighting issue
"""
import json
from pathlib import Path

print("=" * 80)
print("SURAH 1 COMPREHENSIVE DIAGNOSTIC")
print("=" * 80)

# Load data
timing = json.load(open('public/data/abdul_basit/letter_timing_1.json'))
verses = json.load(open('public/data/verses_v4.json'))['1']

# Get Quran text
quran_text = ''.join([v['text'] for v in verses])
quran_chars = [c for c in quran_text if not c.isspace()]

print(f"\n📊 DATA OVERVIEW:")
print(f"  Timing entries: {len(timing)}")
print(f"  Quran non-space chars: {len(quran_chars)}")
print(f"  Match: {'✅ YES' if len(timing) == len(quran_chars) else '❌ NO'}")

print(f"\n📋 TIMING FORMAT:")
if timing:
    first = timing[0]
    print(f"  Fields: {list(first.keys())}")
    print(f"  First entry: {first}")
    print(f"  Time unit: {'seconds' if first.get('start', 0) < 100 else 'milliseconds'}")
    
print(f"\n🎯 CHARACTER ALIGNMENT (first 20):")
for i in range(min(20, len(timing), len(quran_chars))):
    t_char = timing[i].get('char', '?')
    q_char = quran_chars[i]
    t_time = timing[i].get('start', 0)
    match = '✅' if t_char == q_char else '❌'
    print(f"  {i:3d}: timing=\"{t_char}\" ({t_time:6.3f}s) vs quran=\"{q_char}\" {match}")

print(f"\n⏱️  TIMING PROGRESSION:")
print(f"  First letter: '{timing[0]['char']}' at {timing[0]['start']:.3f}s")
print(f"  Last letter:  '{timing[-1]['char']}' at {timing[-1]['start']:.3f}s")
print(f"  Duration: {timing[-1]['end'] - timing[0]['start']:.3f}s")

print(f"\n🔢 WORD INDEX ANALYSIS:")
if 'wordIdx' in timing[0]:
    word_indices = [t.get('wordIdx', -1) for t in timing]
    print(f"  Min wordIdx: {min(word_indices)}")
    print(f"  Max wordIdx: {max(word_indices)}")
    print(f"  Unique words: {len(set(word_indices))}")
    print(f"  Expected words: 29 (Surah 1 has 29 words)")
    
    # Show first few word boundaries
    print(f"\n  First 5 word boundaries:")
    prev_idx = -1
    count = 0
    for i, t in enumerate(timing):
        if t.get('wordIdx') != prev_idx:
            print(f"    Word {t.get('wordIdx')}: starts at char {i} ('{t['char']}') at {t['start']:.3f}s")
            prev_idx = t.get('wordIdx')
            count += 1
            if count >= 5:
                break

print(f"\n📖 QURAN TEXT STRUCTURE:")
for i, v in enumerate(verses):
    words = v['text'].split()
    chars = [c for c in v['text'] if not c.isspace()]
    print(f"  Verse {i+1}: {len(words)} words, {len(chars)} chars")
    print(f"    Text: {v['text'][:50]}...")

print(f"\n🎵 AUDIO EXPECTATIONS:")
print(f"  If audio starts at 0s, first letter should highlight at {timing[0]['start']:.3f}s")
print(f"  If audio has intro, timing offset = {timing[0]['start']:.3f}s")

print(f"\n💡 DIAGNOSIS:")
issues = []
if len(timing) != len(quran_chars):
    issues.append(f"❌ Count mismatch: {len(timing)} timing vs {len(quran_chars)} Quran chars")
else:
    issues.append("✅ Count matches perfectly")

# Check character alignment
mismatches = 0
for i in range(min(len(timing), len(quran_chars))):
    if timing[i].get('char') != quran_chars[i]:
        mismatches += 1

if mismatches > 0:
    issues.append(f"❌ {mismatches} character mismatches")
else:
    issues.append("✅ All characters match")

# Check wordIdx
if 'wordIdx' in timing[0]:
    min_word = min(t.get('wordIdx', 0) for t in timing)
    if min_word != 0:
        issues.append(f"⚠️  wordIdx starts at {min_word}, should be 0")
    else:
        issues.append("✅ wordIdx starts at 0")

# Check timing offset
if timing[0]['start'] > 1:
    issues.append(f"⚠️  Timing starts at {timing[0]['start']:.3f}s (audio may have intro)")
else:
    issues.append(f"✅ Timing starts near 0s ({timing[0]['start']:.3f}s)")

for issue in issues:
    print(f"  {issue}")

print("\n" + "=" * 80)
