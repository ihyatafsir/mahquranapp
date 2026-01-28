#!/usr/bin/env python3
"""
Apply timing offset and drift correction to letter_timing_1.json

Usage: python3 fix_timing_offset.py [offset_ms] [drift_factor]
  offset_ms: Positive = delay start, Negative = earlier start (default: 100)
  drift_factor: Multiplier for accumulated time (default: 1.0, try 1.02 for 2% stretch)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TIMING_PATH = PROJECT_ROOT / "public/data/abdul_basit/letter_timing_1.json"

def apply_correction(offset_ms=100, drift_factor=1.0):
    """
    Apply offset and drift correction.
    
    offset_ms: Shift all timings forward (positive) or backward (negative)
    drift_factor: Stretch time to correct for accumulated drift
    """
    with open(TIMING_PATH, 'r') as f:
        timings = json.load(f)
    
    offset_sec = offset_ms / 1000.0
    
    print(f"Applying correction:")
    print(f"  Offset: {offset_ms}ms ({offset_sec}s)")
    print(f"  Drift factor: {drift_factor}")
    print(f"  Total letters: {len(timings)}")
    
    # Get the first timestamp as reference
    if timings:
        first_start = timings[0]['start']
        print(f"  Original first letter start: {first_start}s")
    
    # Apply corrections
    for t in timings:
        # Apply drift factor first (stretches time from 0)
        t['start'] = t['start'] * drift_factor
        t['end'] = t['end'] * drift_factor
        
        # Then apply offset
        t['start'] += offset_sec
        t['end'] += offset_sec
    
    if timings:
        new_first = timings[0]['start']
        print(f"  New first letter start: {new_first}s")
    
    # Save
    with open(TIMING_PATH, 'w') as f:
        json.dump(timings, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved corrected timings to {TIMING_PATH}")
    
    # Preview
    print("\nFirst 10 letters after correction:")
    for t in timings[:10]:
        dur = (t['end'] - t['start']) * 1000
        print(f"  {t['char']:>2} : {t['start']:.3f}s - {t['end']:.3f}s ({dur:.0f}ms)")


if __name__ == "__main__":
    # Default: add 100ms offset (delay start) to fix "too fast" issue
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    drift = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    apply_correction(offset, drift)
