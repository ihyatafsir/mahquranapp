#!/usr/bin/env python3
"""
Minshawi Mujawwad Acoustic Alignment & Spectrogram Visualizer
High-precision alignment for Sheikh Mohamed Siddiq Al-Minshawi (Long Tahqeeq style).
"""

import json
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIACRITICS = set([
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652',
    '\u0653', '\u0654', '\u0655', '\u0656', '\u0657', '\u0658', '\u065C', '\u065D',
    '\u065E', '\u065F', '\u0670', '\u06E1', '\u06DF', '\u06E0', '\u06E2', '\u06E3'
])
SOLAR_LETTERS = set('تثدذرزسشصضطظلن')

def decompose_minshawi_tajweed(word_str, is_first_in_verse=False, is_ayah_end=False):
    raw_chunks = []
    curr = ""
    for char in word_str:
        if char in DIACRITICS:
            curr += char
        else:
            if curr:
                raw_chunks.append(curr)
            curr = char
    if curr:
        raw_chunks.append(curr)

    processed_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        base_char = ""
        for c in chunk:
            if c not in DIACRITICS:
                base_char = c
                break

        has_maddah = ('\u0653' in chunk) or ('ٓ' in chunk)
        has_shaddah = ('\u0651' in chunk) or ('ّ' in chunk)
        has_sukoon = ('\u0652' in chunk) or ('ْ' in chunk) or ('\u06E1' in chunk)
        has_dagger_alif = ('\u0670' in chunk) or ('ٰ' in chunk)
        is_last = (idx == len(raw_chunks) - 1)
        next_chunk = raw_chunks[idx + 1] if idx + 1 < len(raw_chunks) else ""
        next_base = ""
        for c in next_chunk:
            if c not in DIACRITICS:
                next_base = c
                break

        if base_char == 'ٱ':
            weight = 0.6 if is_first_in_verse else 0.05
        elif base_char == 'ل' and (idx == 1 and raw_chunks[0].startswith('ٱ')) and (next_base in SOLAR_LETTERS):
            weight = 0.05
        elif has_maddah:
            weight = 6.0
        elif is_last and is_ayah_end:
            if base_char in 'قطبجد':
                weight = 3.5  # Waqf Qalqalah
            elif base_char in 'وي' or has_dagger_alif:
                weight = 4.5  # Madd 'Arid
            else:
                weight = 2.5
        elif has_shaddah:
            weight = 2.8 if base_char in 'نم' else 2.2
        elif has_dagger_alif or (base_char in 'اويى' and not has_sukoon and len(chunk) == 1):
            weight = 2.4
        elif not has_sukoon:
            weight = 1.0
        else:
            weight = 1.1 if base_char in 'قطبجد' else (1.2 if base_char in 'سشصضفثذخغحهظز' else 0.7)

        processed_chunks.append({
            'char': chunk,
            'base': base_char,
            'weight': weight
        })
    return processed_chunks

# Align Minshawi Surah 112
audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_112.mp3"
verses_path = "/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json"
output_json = "/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_112.json"
output_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_112_spectrogram.png"

os.makedirs(os.path.dirname(output_json), exist_ok=True)

y, sr = librosa.load(audio_path, sr=22050)
total_dur = len(y) / sr

# Per-Ayah files and offsets
ayah_files = ["/tmp/112001.mp3", "/tmp/112002.mp3", "/tmp/112003.mp3", "/tmp/112004.mp3"]
ayah_durations = [len(librosa.load(f, sr=sr)[0]) / sr for f in ayah_files]
ayah_starts = [0.0]
for d in ayah_durations[:-1]:
    ayah_starts.append(ayah_starts[-1] + d)

print(f"[Minshawi] Total duration: {total_dur:.2f}s")
for i, (s, d) in enumerate(zip(ayah_starts, ayah_durations)):
    print(f"Ayah {i+1}: Start={s:.2f}s, Dur={d:.2f}s, End={s+d:.2f}s")

with open(verses_path, 'r', encoding='utf-8') as f:
    v_list = json.load(f)["112"]

hop_length = 128
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

def snap_time(t, min_t, max_t, win_ms=25):
    w_s = win_ms / 1000.0
    cands = [x for x in onset_times if max(min_t, t - w_s) <= x <= min(max_t, t + w_s)]
    if not cands: return t
    best_c = t
    best_v = -1.0
    for c in cands:
        f = librosa.time_to_frames(c, sr=sr, hop_length=hop_length)
        if f < len(onset_env) and onset_env[f] > best_v:
            best_v = onset_env[f]
            best_c = c
    return float(best_c)

final_letters = []
global_word_idx = 0

for v_idx, verse in enumerate(v_list):
    v_start = ayah_starts[v_idx]
    v_end = v_start + ayah_durations[v_idx]
    words = verse['words']
    
    # Calculate word weights
    all_word_chunks = []
    word_weights = []
    for w_in_v, w in enumerate(words):
        chunks = decompose_minshawi_tajweed(
            w['arabic'],
            is_first_in_verse=(w_in_v == 0),
            is_ayah_end=(w_in_v == len(words) - 1)
        )
        all_word_chunks.append(chunks)
        word_weights.append(sum(c['weight'] for c in chunks))

    total_verse_weight = sum(word_weights)
    cur_word_start = v_start

    for w_in_v, (w, chunks, w_weight) in enumerate(zip(words, all_word_chunks, word_weights)):
        w_dur = ((v_end - v_start) * w_weight) / total_verse_weight
        w_end = (v_end if w_in_v == len(words) - 1 else cur_word_start + w_dur)

        cur_letter_start = cur_word_start
        for c_idx, c_info in enumerate(chunks):
            c_dur = (w_dur * c_info['weight']) / w_weight
            if c_idx == len(chunks) - 1:
                c_end = w_end
            else:
                ideal_end = cur_letter_start + c_dur
                snapped = snap_time(ideal_end, cur_letter_start + 0.05, w_end - 0.05, win_ms=30)
                c_end = min(max(cur_letter_start + 0.05, snapped), w_end - 0.05)

            final_letters.append({
                'charIdx': len(final_letters),
                'char': c_info['char'],
                'start': round(cur_letter_start, 3),
                'end': round(c_end, 3),
                'duration': round(c_end - cur_letter_start, 3),
                'wordIdx': global_word_idx,
                'ayah': verse['ayah'],
                'verseIdx': v_idx
            })
            cur_letter_start = c_end

        cur_word_start = w_end
        global_word_idx += 1

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_letters, f, ensure_ascii=False, indent=2)

print(f"\n[Minshawi] Successfully generated {len(final_letters)} letters!")

# Plot High-Resolution Spectrogram
time_axis = np.linspace(0, total_dur, len(y))
fig, axes = plt.subplots(3, 1, figsize=(24, 11), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.3]})

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                              hop_length=hop_length, ax=axes[0], cmap='inferno')
axes[0].set_title("Sheikh Al-Minshawi (Mujawwad Tahqeeq Style) - High-Precision Spectrogram (Surah 112)", fontsize=14, fontweight='bold')
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

axes[1].plot(time_axis, y, color='#00ff88', alpha=0.6, label='Audio Waveform')
onset_times_full = np.linspace(0, total_dur, len(onset_env))
axes[1].plot(onset_times_full, onset_env / (np.max(onset_env) + 1e-6) * np.max(np.abs(y)), color='#00f0ff', lw=2, label='Transient Energy E(t)')
axes[1].set_title("Acoustic Transient Releases & Qalqalah Onsets", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.2)

axes[2].set_ylim(0, 1)
axes[2].set_yticks([])
axes[2].set_title("Aligned Letters with Uthmani Diacritics & Sibawayh Proportional Duration", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12']

for l in final_letters:
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
print(f"[Visualizer] Saved Minshawi Mujawwad visual spectrogram to: {output_plot}")
