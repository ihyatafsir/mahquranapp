#!/usr/bin/env python3
"""
Pristine Master Audio & Letter Alignment for Sheikh Al-Minshawi (Surah 1: Al-Fatiha)
Resolves all Reciter Repetition (Takrar) and Multi-Ayah Wasl:
- Ayah 1 (001001.mp3): 4.40s - 11.70s -> [بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ]
- Ayah 2 (001002.mp3): 4.41s - 10.82s -> [ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ]
- Ayah 3 (001003.mp3): 3.96s - 16.36s -> [ٱلرَّحْمَٰنِ ٱلرَّحِيمِ]
- Ayah 4 (001004.mp3): 17.50s - 23.85s -> [مَٰلِكِ يَوْمِ ٱلدِّينِ] (Repeated Ayahs 2 & 3 cleanly trimmed!)
- Ayah 5 (001005.mp3): 5.38s - 13.84s -> [إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ]
- Ayah 6 (001006.mp3): 4.90s - 10.87s -> [ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ]
- Ayah 7 (001007.mp3): 0.05s - 13.51s (Part A) + 22.48s - 34.85s (Part B)
"""

import json
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sibawayh_acoustic_aligner import chunk_arabic_word

with open("/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json", 'r', encoding='utf-8') as f:
    verses_data = json.load(f)["1"]

# Definitions of each clean master Ayah slice from raw EveryAyah files
# Format: (ayah_num, raw_file_path, [(word_start, word_end, slice_start_s, slice_end_s)])
MASTER_SLICES = [
    (1, "/tmp/minshawi_001/001001.mp3", [(0, 4, 4.40, 11.70)]),
    (2, "/tmp/minshawi_001/001002.mp3", [(0, 4, 4.41, 10.82)]),
    (3, "/tmp/minshawi_001/001003.mp3", [(0, 2, 3.96, 16.36)]),
    # Ayah 4: Slice directly to where مَٰلِكِ يَوْمِ ٱلدِّينِ begins (17.5s - 23.85s)!
    (4, "/tmp/minshawi_001/001004.mp3", [(0, 3, 17.50, 23.85)]),
    (5, "/tmp/minshawi_001/001005.mp3", [(0, 4, 5.38, 13.84)]),
    (6, "/tmp/minshawi_001/001006.mp3", [(0, 3, 4.90, 10.87)]),
    # Ayah 7: Part A (0.05 - 13.51) + 1.0s breath pause + Part B (22.48 - 34.85)
    (7, "/tmp/minshawi_001/001007.mp3", [
        (0, 4, 0.05, 13.51),
        (4, 9, 22.48, 34.85)
    ])
]

sr = 22050
master_audio_clips = []
final_letters = []
global_word_idx = 0
cur_time_offset = 0.0

# Tajweed Duration Weight Rules
def get_letter_weight(chunk, is_word_last, is_ayah_last, base_char):
    has_maddah = ('ٓ' in chunk) or ('\u0653' in chunk)
    has_shaddah = ('ّ' in chunk) or ('\u0651' in chunk)
    has_dagger_alif = ('ٰ' in chunk) or ('\u0670' in chunk)
    has_sukoon = ('ْ' in chunk) or ('\u0652' in chunk) or ('\u06E1' in chunk)

    if has_maddah: return 6.0
    if is_ayah_last and is_word_last and (base_char in 'وي' or has_dagger_alif): return 4.5
    if has_shaddah and base_char in 'نم': return 2.8
    if has_shaddah: return 2.0
    if has_dagger_alif or (base_char in 'اوي' and not has_sukoon and len(chunk) == 1): return 2.2
    if has_sukoon: return 0.8
    return 1.0

# 0.4s clean natural silence between Ayahs
inter_ayah_silence = np.zeros(int(0.4 * sr))

for ayah_num, raw_path, parts in MASTER_SLICES:
    ayah_idx = ayah_num - 1
    y_raw, _ = librosa.load(raw_path, sr=sr)
    
    verse_info = verses_data[ayah_idx]
    all_words = verse_info['words']

    print(f"\n=== Processing Master Ayah {ayah_num} ===")

    for part_idx, (w_s_idx, w_e_idx, slice_start, slice_end) in enumerate(parts):
        # Extract audio slice
        i_s = int(slice_start * sr)
        i_e = int(slice_end * sr)
        audio_slice = y_raw[i_s:i_e]
        slice_dur = len(audio_slice) / sr

        seg_words = all_words[w_s_idx:w_e_idx]

        # Calculate word weights
        seg_word_weights = []
        for w in seg_words:
            chunks = chunk_arabic_word(w['arabic'])
            w_w = sum(get_letter_weight(c, i == len(chunks)-1, False, c[0]) for i, c in enumerate(chunks))
            seg_word_weights.append(w_w)

        tot_w_weight = sum(seg_word_weights)
        cur_w_start = cur_time_offset

        for w_rel_idx, w in enumerate(seg_words):
            w_abs_idx = w_s_idx + w_rel_idx
            is_ayah_end_word = (w_abs_idx == len(all_words) - 1)
            is_seg_end_word = (w_rel_idx == len(seg_words) - 1)

            w_dur = (slice_dur * seg_word_weights[w_rel_idx]) / tot_w_weight
            w_end = cur_w_start + w_dur

            chunks = chunk_arabic_word(w['arabic'])
            letter_weights = [
                get_letter_weight(c, i == len(chunks)-1, is_ayah_end_word or is_seg_end_word, c[0])
                for i, c in enumerate(chunks)
            ]
            tot_l_weight = sum(letter_weights)

            cur_l_start = cur_w_start
            for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
                l_dur = (w_dur * lw) / tot_l_weight
                l_e = w_end if l_idx == len(chunks) - 1 else cur_l_start + l_dur

                final_letters.append({
                    'charIdx': len(final_letters),
                    'char': chunk,
                    'start': round(cur_l_start, 3),
                    'end': round(l_e, 3),
                    'duration': round(l_e - cur_l_start, 3),
                    'wordIdx': global_word_idx,
                    'ayah': ayah_num,
                    'verseIdx': ayah_idx
                })
                cur_l_start = l_e

            print(f"  Word #{global_word_idx} '{w['arabic']}': {cur_w_start:.3f}s - {w_end:.3f}s (dur: {w_dur:.3f}s)")
            cur_w_start = w_end
            global_word_idx += 1

        master_audio_clips.append(audio_slice)
        cur_time_offset += slice_dur

        # If Ayah 7 Part A, insert 0.8s breath pause
        if ayah_num == 7 and part_idx == 0:
            pause_silence = np.zeros(int(0.8 * sr))
            master_audio_clips.append(pause_silence)
            cur_time_offset += 0.8

    # Insert inter-ayah pause
    if ayah_num < 7:
        master_audio_clips.append(inter_ayah_silence)
        cur_time_offset += 0.4

# Save continuous master audio
y_full_master = np.concatenate(master_audio_clips)
master_audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3"
sf.write(master_audio_path, y_full_master, sr)
print(f"\n[Master Audio] Exported clean master track: {master_audio_path} (Duration: {len(y_full_master)/sr:.2f}s)")

output_json = "/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_letters, f, ensure_ascii=False, indent=2)
print(f"[Timing JSON] Exported {len(final_letters)} letters to: {output_json}")

# Visual Spectrogram
fig, axes = plt.subplots(2, 1, figsize=(28, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
S = librosa.feature.melspectrogram(y=y_full_master, sr=sr, n_mels=128, fmax=8000, hop_length=256)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                              hop_length=256, ax=axes[0], cmap='inferno')
axes[0].set_title("Sheikh Al-Minshawi (Surah 1: Al-Fatiha) - Pristine Master Single-Take Alignment", fontsize=14, fontweight='bold')
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

axes[1].set_ylim(0, 1)
axes[1].set_yticks([])
axes[1].set_title("Synchronized Letter Karaoke Stream", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12', '#064e3b']
for l in final_letters:
    dur = l['end'] - l['start']
    if dur <= 0: continue
    col = colors[l['wordIdx'] % len(colors)]
    rect = plt.Rectangle((l['start'], 0.1), dur, 0.8, color=col, alpha=0.85, ec='#38bdf8', lw=1.2)
    axes[1].add_patch(rect)
    mid_x = (l['start'] + l['end']) / 2
    axes[1].text(mid_x, 0.5, l['char'], fontsize=8.5, color='#ffffff', ha='center', va='center', fontweight='bold', fontname='DejaVu Sans')

plt.tight_layout()
out_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_001_pristine_master.png"
plt.savefig(out_plot, dpi=200)
print(f"[Visualizer] Plot saved to: {out_plot}")
