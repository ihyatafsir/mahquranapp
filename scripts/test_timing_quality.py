#!/usr/bin/env python3
"""
Timing Quality Test Suite
Tests timing data quality for accurate letter highlighting.
"""
import json
from pathlib import Path
import statistics

DATA_DIR = Path(__file__).parent.parent / "public" / "data" / "abdul_basit"
VERSES_FILE = Path(__file__).parent.parent / "public" / "data" / "verses_v4.json"

def load_verses():
    with open(VERSES_FILE) as f:
        return json.load(f)

def test_surah(surah_num, verses, timing_data):
    """Run all quality tests on a surah's timing data."""
    results = {
        "surah": surah_num,
        "total_chars": len(timing_data),
        "issues": []
    }
    
    # Test 1: Coverage - all timing entries should have non-zero timing
    zero_timing = [t for t in timing_data if t.get("start", 0) == 0 and t.get("end", 0) == 0]
    coverage = 1 - (len(zero_timing) / len(timing_data)) if timing_data else 0
    results["coverage"] = round(coverage * 100, 1)
    if coverage < 0.9:
        results["issues"].append(f"Low coverage: {coverage*100:.1f}% (expected >90%)")
    
    # Test 2: Monotonicity - timing should generally increase
    monotonic_errors = 0
    for i in range(1, len(timing_data)):
        prev_end = timing_data[i-1].get("end", 0) or 0
        curr_start = timing_data[i].get("start", 0) or 0
        if curr_start > 0 and prev_end > 0 and curr_start < prev_end - 0.1:
            monotonic_errors += 1
    results["monotonic_errors"] = monotonic_errors
    if monotonic_errors > 5:
        results["issues"].append(f"Monotonicity issues: {monotonic_errors}")
    
    # Test 3: Duration reasonableness - most letters should be 10-500ms
    durations = [t.get("duration", 0) or 0 for t in timing_data if t.get("duration")]
    if durations:
        # Convert to ms if in seconds
        if max(durations) < 10:  # Likely seconds
            durations = [d * 1000 for d in durations]
        
        short_count = sum(1 for d in durations if d < 5)
        long_count = sum(1 for d in durations if d > 500)
        results["avg_duration_ms"] = round(statistics.mean(durations), 1)
        results["short_durations"] = short_count
        results["long_durations"] = long_count
    
    # Test 4: Gap detection - large gaps between letters
    gaps = []
    for i in range(1, len(timing_data)):
        prev_end = timing_data[i-1].get("end", 0) or 0
        curr_start = timing_data[i].get("start", 0) or 0
        if prev_end > 0 and curr_start > 0:
            gap = curr_start - prev_end
            if gap > 0.5:  # 500ms gap
                gaps.append(gap)
    results["large_gaps"] = len(gaps)
    
    # Test 5: Character count match
    verse_text = ''.join(v['text'] for v in verses)
    expected_non_space = len([c for c in verse_text if not c.isspace()])
    actual_count = len(timing_data)
    match_pct = (actual_count / expected_non_space * 100) if expected_non_space else 0
    results["char_count_match"] = round(match_pct, 1)
    if abs(match_pct - 100) > 5:
        results["issues"].append(f"Character count mismatch: {actual_count} vs {expected_non_space}")
    
    # Overall score
    score = 100
    score -= (1 - coverage) * 30  # -30 for 0% coverage
    score -= min(monotonic_errors, 10) * 2  # -2 per error, max -20
    score -= len(results.get("issues", [])) * 5
    results["quality_score"] = max(0, round(score, 1))
    
    return results

def main():
    print("=" * 60)
    print("TIMING QUALITY TEST SUITE")
    print("=" * 60)
    
    all_verses = load_verses()
    
    results = []
    for timing_file in sorted(DATA_DIR.glob("letter_timing_*.json")):
        surah_num = int(timing_file.stem.split('_')[-1])
        verses = all_verses.get(str(surah_num), [])
        
        if not verses:
            continue
        
        with open(timing_file) as f:
            timing_data = json.load(f)
        
        result = test_surah(surah_num, verses, timing_data)
        results.append(result)
        
        # Print summary
        status = "✓" if result["quality_score"] >= 80 else "⚠" if result["quality_score"] >= 60 else "✗"
        print(f"{status} Surah {surah_num:3d}: Score={result['quality_score']:5.1f}  "
              f"Coverage={result['coverage']:5.1f}%  "
              f"Errors={result['monotonic_errors']:2d}  "
              f"Chars={result['total_chars']}")
        
        if result["issues"]:
            for issue in result["issues"]:
                print(f"    → {issue}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        avg_score = statistics.mean(r["quality_score"] for r in results)
        avg_coverage = statistics.mean(r["coverage"] for r in results)
        total_errors = sum(r["monotonic_errors"] for r in results)
        
        print(f"Total surahs tested: {len(results)}")
        print(f"Average quality score: {avg_score:.1f}")
        print(f"Average coverage: {avg_coverage:.1f}%")
        print(f"Total monotonicity errors: {total_errors}")
        
        low_quality = [r for r in results if r["quality_score"] < 70]
        if low_quality:
            print(f"\nLow quality surahs ({len(low_quality)}):")
            for r in low_quality:
                print(f"  - Surah {r['surah']}: Score={r['quality_score']}")

if __name__ == "__main__":
    main()
