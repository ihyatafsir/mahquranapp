#!/usr/bin/env python3
"""
Alignment Test Tool - Simulate App.tsx highlighting logic without browser

This tool:
1. Loads Quran text and timing data
2. Simulates the highlighting logic from App.tsx
3. Tests alignment approaches
4. Reports accuracy metrics
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

DIACRITICS = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u065C\u065D\u065E\u065F\u0670'

def is_diacritic(c): return c in DIACRITICS
def is_space(c): return c.isspace()

class AlignmentTester:
    def __init__(self, base_path: str = 'public/data'):
        self.base_path = Path(base_path)
        self.verses_path = self.base_path / 'verses_v4.json'
        self.all_verses = json.load(open(self.verses_path, 'r', encoding='utf-8'))
    
    def load_timing(self, surah: int, reciter: str = 'abdul_basit') -> List[Dict]:
        """Load timing data for a surah."""
        timing_path = self.base_path / reciter / f'letter_timing_{surah}.json'
        if not timing_path.exists():
            return []
        data = json.load(open(timing_path, 'r', encoding='utf-8'))
        return [d for d in data if d.get('char', '') != '\ufeff']
    
    def get_quran_chars(self, surah: int) -> List[str]:
        """Get all non-space chars from Quran text."""
        verses = self.all_verses.get(str(surah), [])
        text = ''.join([v['text'] for v in verses])
        return [c for c in text if not is_space(c)]
    
    def get_quran_base_letters(self, surah: int) -> List[str]:
        """Get base letters only (no diacritics)."""
        chars = self.get_quran_chars(surah)
        return [c for c in chars if not is_diacritic(c)]
    
    def build_char_groups(self, surah: int) -> List[Dict]:
        """
        Build character groups like App.tsx does:
        Each group = base letter + following diacritics
        """
        verses = self.all_verses.get(str(surah), [])
        text = ''.join([v['text'] for v in verses])
        
        groups = []
        current_group = {'chars': '', 'base_idx': 0}
        base_idx = 0
        
        for c in text:
            if is_space(c):
                if current_group['chars']:
                    groups.append(current_group.copy())
                    current_group = {'chars': '', 'base_idx': base_idx}
            elif is_diacritic(c):
                current_group['chars'] += c
            else:
                if current_group['chars']:
                    groups.append(current_group.copy())
                current_group = {'chars': c, 'base_idx': base_idx}
                base_idx += 1
        
        if current_group['chars']:
            groups.append(current_group)
        
        return groups
    
    def normalize_timing(self, timing: List[Dict]) -> List[Dict]:
        """Normalize timing to seconds if needed."""
        if not timing:
            return timing
        first_start = timing[0].get('start', 0)
        if first_start > 100:  # Milliseconds
            return [{**t, 
                     'start': t['start'] / 1000,
                     'end': t['end'] / 1000,
                     'duration': t.get('duration', 0) / 1000} 
                    for t in timing]
        return timing
    
    def test_alignment(self, surah: int, reciter: str = 'abdul_basit') -> Dict:
        """
        Test alignment between timing data and Quran text.
        Returns detailed metrics.
        """
        timing = self.load_timing(surah, reciter)
        timing = self.normalize_timing(timing)
        
        quran_all_chars = self.get_quran_chars(surah)
        quran_base = self.get_quran_base_letters(surah)
        groups = self.build_char_groups(surah)
        
        result = {
            'surah': surah,
            'timing_entries': len(timing),
            'quran_all_chars': len(quran_all_chars),
            'quran_base_letters': len(quran_base),
            'char_groups': len(groups),
            'issues': []
        }
        
        if not timing:
            result['issues'].append('No timing data found')
            return result
        
        # Check timing format
        first = timing[0]
        result['timing_format'] = {
            'has_charIdx': 'charIdx' in first,
            'has_idx': 'idx' in first,
            'has_wordIdx': 'wordIdx' in first,
            'has_ayah': 'ayah' in first,
            'time_unit': 'ms' if first.get('start', 0) > 100 else 'seconds',
            'first_char': first.get('char', '?'),
            'first_start': first.get('start', 0)
        }
        
        # Approach 1: Timing includes diacritics (matches all chars)
        if len(timing) == len(quran_all_chars):
            result['alignment_type'] = 'FULL_CHARS'
            result['match_score'] = self._calc_char_match(timing, quran_all_chars)
        # Approach 2: Timing is base letters only (matches base)
        elif len(timing) == len(quran_base):
            result['alignment_type'] = 'BASE_ONLY'
            result['match_score'] = self._calc_char_match(timing, quran_base)
        # Approach 3: Timing matches char groups
        elif len(timing) == len(groups):
            result['alignment_type'] = 'GROUPS'
            result['match_score'] = 'N/A'
        else:
            result['alignment_type'] = 'MISMATCH'
            result['issues'].append(
                f'Count mismatch: timing={len(timing)}, all_chars={len(quran_all_chars)}, '
                f'base={len(quran_base)}, groups={len(groups)}'
            )
        
        # Test specific time points
        result['time_samples'] = self._sample_times(timing)
        
        return result
    
    def _calc_char_match(self, timing: List[Dict], quran_chars: List[str]) -> float:
        """Calculate percentage of matching characters."""
        matches = 0
        for i in range(min(len(timing), len(quran_chars))):
            t_char = timing[i].get('char', '')
            q_char = quran_chars[i]
            if self._chars_match(t_char, q_char):
                matches += 1
        return matches / max(len(timing), len(quran_chars)) * 100
    
    def _chars_match(self, c1: str, c2: str) -> bool:
        """Check if two chars match (allowing variants)."""
        if c1 == c2:
            return True
        variants = {'ٱ': 'ا', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا'}
        return variants.get(c1, c1) == variants.get(c2, c2)
    
    def _sample_times(self, timing: List[Dict]) -> List[Dict]:
        """Sample timing at different points."""
        if len(timing) < 5:
            return []
        indices = [0, len(timing)//4, len(timing)//2, 3*len(timing)//4, len(timing)-1]
        return [{
            'idx': i,
            'char': timing[i].get('char', '?'),
            'start': timing[i].get('start', 0),
            'end': timing[i].get('end', 0)
        } for i in indices]
    
    def run_batch_test(self, surahs: List[int] = None, reciter: str = 'abdul_basit'):
        """Run test on multiple surahs."""
        if surahs is None:
            surahs = range(1, 115)
        
        results = []
        for s in surahs:
            r = self.test_alignment(s, reciter)
            results.append(r)
            
            status = '✅' if r.get('alignment_type') in ['FULL_CHARS', 'BASE_ONLY'] and \
                           r.get('match_score', 0) > 90 else \
                     '⚠️' if r.get('alignment_type') == 'MISMATCH' else '❌'
            
            print(f"Surah {s:3d}: {status} {r.get('alignment_type', 'UNKNOWN'):12s} "
                  f"timing={r['timing_entries']:5d} quran={r['quran_all_chars']:5d} "
                  f"match={r.get('match_score', 'N/A')}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test timing alignment')
    parser.add_argument('--surah', type=int, help='Single surah to test')
    parser.add_argument('--all', action='store_true', help='Test all surahs')
    parser.add_argument('--reciter', default='abdul_basit', help='Reciter folder')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    tester = AlignmentTester()
    
    if args.surah:
        result = tester.test_alignment(args.surah, args.reciter)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.all:
        tester.run_batch_test(reciter=args.reciter)
    else:
        # Default: test sample surahs
        tester.run_batch_test([1, 36, 55, 67, 84, 99, 112, 114], args.reciter)


if __name__ == '__main__':
    main()
