import json
import os

DATA_DIR = "/home/grem3/mahquranapp/public/data"
WAVEFORM_PATH = os.path.join(DATA_DIR, "waveforms", "abdul_basit_murattal_surah_1.json")
TIMING_PATH = os.path.join(DATA_DIR, "abdul_basit_murattal", "letter_timing_1.json")

with open(WAVEFORM_PATH, "r", encoding="utf-8") as f:
    wf = json.load(f)
peaks = wf["peaks"]
pps = wf["peaksPerSecond"] # 100

with open(TIMING_PATH, "r", encoding="utf-8") as f:
    timing = json.load(f)

# Find local maximum peak inside each letter window [start, end]
for item in timing:
    s = item["start"]
    e = item["end"]
    s_idx = max(0, int(s * pps))
    e_idx = min(len(peaks), max(s_idx + 1, int(e * pps)))
    
    letter_peaks = peaks[s_idx:e_idx]
    if letter_peaks:
        max_val = max(letter_peaks)
        max_local_idx = letter_peaks.index(max_val)
        peak_time = round(s + (max_local_idx / pps), 3)
    else:
        max_val = 0.1
        peak_time = round((s + e) / 2, 3)
        
    item["peakTime"] = peak_time
    item["peakEnergy"] = max_val

with open(TIMING_PATH, "w", encoding="utf-8") as f:
    json.dump(timing, f, ensure_ascii=False, indent=2)

print(f"Locked {len(timing)} letters directly to their true acoustic waveform peaks in {TIMING_PATH}!")
