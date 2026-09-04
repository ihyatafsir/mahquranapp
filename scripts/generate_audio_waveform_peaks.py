import json
import os
import subprocess
import struct

AUDIO_PATH = "/home/grem3/mahquranapp/public/audio/abdul_basit_murattal/surah_001.mp3"
WAVEFORM_DIR = "/home/grem3/mahquranapp/public/data/waveforms"
os.makedirs(WAVEFORM_DIR, exist_ok=True)
OUT_PATH = os.path.join(WAVEFORM_DIR, "abdul_basit_murattal_surah_1.json")

# Decode MP3 to raw 16kHz mono PCM via ffmpeg
cmd = [
    "ffmpeg", "-y", "-i", AUDIO_PATH,
    "-ac", "1", "-ar", "16000", "-f", "s16le", "-"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw_pcm, _ = proc.communicate()

# Convert raw PCM bytes to 16-bit integers
samples = struct.unpack(f"{len(raw_pcm)//2}h", raw_pcm)
sample_rate = 16000
duration = len(samples) / sample_rate

# Compute 100 peaks per second (160 samples per window = 10ms)
window_size = 160
peaks = []
rms_values = []

for i in range(0, len(samples), window_size):
    chunk = samples[i:i+window_size]
    if not chunk:
        continue
    peak = max(abs(s) for s in chunk) / 32768.0
    rms = (sum(s*s for s in chunk) / len(chunk))**0.5 / 32768.0
    peaks.append(round(peak, 4))
    rms_values.append(round(rms, 4))

waveform_data = {
    "reciter": "abdul_basit_murattal",
    "surah": 1,
    "duration": round(duration, 3),
    "sampleRate": sample_rate,
    "peaksPerSecond": 100,
    "peaks": peaks,
    "rms": rms_values
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(waveform_data, f)

print(f"Generated high-resolution waveform peaks: {len(peaks)} peaks across {duration:.2f}s ({OUT_PATH})")
