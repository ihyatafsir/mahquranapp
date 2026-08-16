#!/usr/bin/env python3
"""
Waveform & Spectrogram Visual Alignment Inspector
Generates high-resolution spectrogram + waveform plots with overlaid letter boundaries.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# Load audio snippet
audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/surah_036.mp3"
letters_path = "/home/absolut7/Documents/mahquranapp/public/data/letter_timing_36.json"
output_image = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/surah36_waveform_alignment.png"

# Inspect Ayah 1, 2, 3 (Time range 7.8s to 19.5s)
T_START = 7.8
T_END = 19.5

y, sr = librosa.load(audio_path, sr=22050, offset=T_START, duration=(T_END - T_START))
time_axis = np.linspace(T_START, T_END, len(y))

# Load letters in range
with open(letters_path, 'r', encoding='utf-8') as f:
    all_letters = json.load(f)

letters_in_range = [l for l in all_letters if l['start'] >= T_START - 0.5 and l['end'] <= T_END + 0.5]

# Create high-res 3-panel visual plot
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.2]})

# Panel 1: Log Mel-Spectrogram with Formants & Harmonic Energy
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=256)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                              hop_length=256, ax=axes[0], x_coords=np.linspace(T_START, T_END, S_dB.shape[1]), cmap='viridis')
axes[0].set_title("1. Mel-Spectrogram (Formants, Vowel Bands & Madd Ridges)", fontsize=13, fontweight='bold', pad=10)
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

# Panel 2: Audio Waveform & Onset Energy Envelope
axes[1].plot(time_axis, y, color='#00ff88', alpha=0.6, label='Waveform')
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
onset_times = np.linspace(T_START, T_END, len(onset_env))
axes[1].plot(onset_times, onset_env / (np.max(onset_env) + 1e-6) * np.max(np.abs(y)), 
             color='#ff007f', lw=2, label='Transient Onset Energy E(t)')
axes[1].set_ylabel("Amplitude", fontsize=11)
axes[1].set_title("2. Audio Waveform & Acoustic Transient Spikes", fontsize=13, fontweight='bold', pad=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.2)

# Panel 3: Letter & Diacritic Boundaries Banner
axes[2].set_ylim(0, 1)
axes[2].set_yticks([])
axes[2].set_title("3. Aligned Uthmani Letters & Tajweed Duration Blocks", fontsize=13, fontweight='bold', pad=10)
axes[2].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12']

for idx, l in enumerate(letters_in_range):
    start = max(T_START, l['start'])
    end = min(T_END, l['end'])
    dur = end - start
    if dur <= 0: continue
    
    col = colors[l['wordIdx'] % len(colors)]
    rect = plt.Rectangle((start, 0.1), dur, 0.8, color=col, alpha=0.85, ec='#38bdf8', lw=1.2)
    axes[2].add_patch(rect)
    
    # Label Arabic character
    mid_x = (start + end) / 2
    axes[2].text(mid_x, 0.5, l['char'], fontsize=14, color='#ffffff', 
                 ha='center', va='center', fontweight='bold', fontname='DejaVu Sans')
    
    # Draw vertical boundary line across all panels
    for ax in axes:
        ax.axvline(x=l['start'], color='#38bdf8', linestyle='--', alpha=0.5, lw=0.8)

plt.tight_layout()
plt.savefig(output_image, dpi=200)
print(f"[Visualizer] High-resolution waveform inspection plot saved to: {output_image}")
