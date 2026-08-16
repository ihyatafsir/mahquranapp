#!/usr/bin/env python3
"""
Test 2-Stage Sibawayh Acoustic Physics Alignment on Surah 112 (Al-Ikhlas)
Generates sub-millisecond letter timing + full-surah visual spectrogram.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# Files
audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/surah_112.mp3"
verses_path = "/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json"
timing_path = "/home/absolut7/Documents/26apps/ihyatafsir-android/assets/audio_mah/timing_112.json"
verse_timing_path = "/home/absolut7/Documents/26apps/ihyatafsir-android/assets/audio_mah/verse_timing_112.json"
output_json = "/home/absolut7/Documents/mahquranapp/public/data/letter_timing_112.json"
output_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/surah112_full_waveform_alignment.png"

# Import Sibawayh Audio Aligner
from sibawayh_acoustic_aligner import SibawayhAudioAligner, chunk_arabic_word

aligner = SibawayhAudioAligner(audio_path)
letters = aligner.align_surah(112, verses_path, timing_path, verse_timing_path)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(letters, f, ensure_ascii=False, indent=2)

print(f"\n[Sibawayh 112] Generated {len(letters)} letters across {letters[-1]['ayah']} Ayahs!")
for l in letters:
    print(f"Ayah {l['ayah']} | Word #{l['wordIdx']} | '{l['char']:<4}' | {l['start']:.3f}s - {l['end']:.3f}s (dur: {l['duration']:.3f}s)")

# Generate full-surah high-res visual spectrogram
y, sr = librosa.load(audio_path, sr=22050)
time_axis = np.linspace(0, len(y)/sr, len(y))

fig, axes = plt.subplots(3, 1, figsize=(20, 11), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.3]})

# 1. Mel-Spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=256)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, hop_length=256, ax=axes[0], cmap='magma')
axes[0].set_title("Surah 112 (Al-Ikhlas) - Mel-Spectrogram (Formant Tracks, Qalqalah Bursts & Vowel Ridges)", fontsize=13, fontweight='bold')
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

# 2. Waveform & Transient Spikes
axes[1].plot(time_axis, y, color='#00ff88', alpha=0.6, label='Audio Waveform')
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
onset_times = np.linspace(0, len(y)/sr, len(onset_env))
axes[1].plot(onset_times, onset_env / (np.max(onset_env) + 1e-6) * np.max(np.abs(y)), color='#00f0ff', lw=2, label='Transient Onset Energy E(t)')
axes[1].set_title("Acoustic Transient Release Spikes (Qaf, Dal Qalqalah, Lam Sukoon)", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.2)

# 3. Aligned Letters
axes[2].set_ylim(0, 1)
axes[2].set_yticks([])
axes[2].set_title("Aligned Letters with Uthmani Diacritics & Sibawayh Proportional Duration", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12']

for idx, l in enumerate(letters):
    dur = l['end'] - l['start']
    if dur <= 0: continue
    col = colors[l['wordIdx'] % len(colors)]
    rect = plt.Rectangle((l['start'], 0.1), dur, 0.8, color=col, alpha=0.85, ec='#38bdf8', lw=1.2)
    axes[2].add_patch(rect)
    
    mid_x = (l['start'] + l['end']) / 2
    axes[2].text(mid_x, 0.5, l['char'], fontsize=11, color='#ffffff', ha='center', va='center', fontweight='bold', fontname='DejaVu Sans')
    
    for ax in axes:
        ax.axvline(x=l['start'], color='#38bdf8', linestyle='--', alpha=0.4, lw=0.6)

plt.tight_layout()
plt.savefig(output_plot, dpi=200)
print(f"\n[Visualizer] Saved full Surah 112 visual spectrogram plot to: {output_plot}")
