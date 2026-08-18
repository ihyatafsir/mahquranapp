#!/usr/bin/env python3
"""
True Quranic Waqf & Takrar-Aware Alignment for Sheikh Al-Minshawi (Surah 1: Al-Fatiha)
Corrects the intermediate Waqf pause in Ayah 7:
- Part A (Words 0-3: صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ) -> Segment 1 (0.05s - 13.51s)
- Pause (13.51s - 22.48s) -> Stationary on عَلَيْهِمْ
- Part B (Words 4-8: غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ) -> Segment 2 (22.48s - 34.85s)
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

# Exact physical speech intervals per Ayah in raw EveryAyah files
# Format: (ayah_num, [(word_start_idx, word_end_idx, voice_start_s, voice_end_s)])
AYAH_SEGMENTS = [
    # Ayah 1 (All 4 words)
    (1, [(0, 4, 4.40, 11.70)]),
    # Ayah 2 (All 4 words)
    (2, [(0, 4, 4.41, 10.82)]),
    # Ayah 3 (All 2 words)
    (3, [(0, 2, 3.96, 16.36)]),
    # Ayah 4 (All 3 words)
    (4, [(0, 3, 5.32, 23.85)]),
    # Ayah 5 (All 4 words)
    (5, [(0, 4, 5.38, 13.84)]),
    # Ayah 6 (All 3 words)
    (6, [(0, 3, 4.90, 10.87)]),
    # Ayah 7: Split at Waqf on عَلَيْهِمْ
    # Part A: صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ (Words 0-3) in Seg 1
    # Part B: غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ (Words 4-8) in Seg 2
    (7, [
        (0, 4, 0.05, 13.51),
        (4, 9, 22.48, 34.85)
    ])
]

sr = 22050
ayah_clips = []
final_letters = []
global_word_idx = 0
cur_surah_offset = 0.0

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

for ayah_num, segs in AYAH_SEGMENTS:
    ayah_idx = ayah_num - 1
    raw_path = f"/tmp/minshawi_001/00100{ayah_num}.mp3"
    y_raw, _ = librosa.load(raw_path, sr=sr)
    raw_dur = len(y_raw) / sr

    verse_info = verses_data[ayah_idx]
    all_words = verse_info['words']

    print(f"\n==================== AYAH {ayah_num} (Offset: {cur_surah_offset:.2f}s, Dur: {raw_dur:.2f}s) ====================")

    for seg_idx, (w_s_idx, w_e_idx, v_s, v_e) in enumerate(segs):
        seg_words = all_words[w_s_idx:w_e_idx]
        seg_dur = v_e - v_s

        # Calculate word weights in this segment
        seg_word_weights = []
        for w in seg_words:
            chunks = chunk_arabic_word(w['arabic'])
            w_w = sum(get_letter_weight(c, i == len(chunks)-1, False, c[0]) for i, c in enumerate(chunks))
            seg_word_weights.append(w_w)

        tot_w_weight = sum(seg_word_weights)
        cur_w_start = cur_surah_offset + v_s

        for w_rel_idx, w in enumerate(seg_words):
            w_abs_idx = w_s_idx + w_rel_idx
            is_ayah_end_word = (w_abs_idx == len(all_words) - 1)
            is_seg_end_word = (w_rel_idx == len(seg_words) - 1)

            w_dur = (seg_dur * seg_word_weights[w_rel_idx]) / tot_w_weight
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
                l_end = w_end if l_idx == len(chunks) - 1 else cur_l_start + l_dur

                final_letters.append({
                    'charIdx': len(final_letters),
                    'char': chunk,
                    'start': round(cur_l_start, 3),
                    'end': round(l_end, 3),
                    'duration': round(l_end - cur_l_start, 3),
                    'wordIdx': global_word_idx,
                    'ayah': ayah_num,
                    'verseIdx': ayah_idx
                })
                cur_l_start = l_end

            print(f"  Word #{global_word_idx} '{w['arabic']}': {cur_w_start:.3f}s - {w_end:.3f}s (dur: {w_dur:.3f}s, letters: {len(chunks)})")
            cur_w_start = w_end
            global_word_idx += 1

    ayah_clips.append(y_raw)
    cur_surah_offset += raw_dur

# Save master continuous audio
y_master = np.concatenate(ayah_clips)
master_audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3"
sf.write(master_audio_path, y_master, sr)
print(f"\n[Master Audio] Exported: {master_audio_path} (Duration: {len(y_master)/sr:.2f}s)")

output_json = "/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_letters, f, ensure_ascii=False, indent=2)
print(f"[Timing JSON] Exported {len(final_letters)} letters to: {output_json}")

# Visual Spectrogram
fig, axes = plt.subplots(2, 1, figsize=(28, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
S = librosa.feature.melspectrogram(y=y_master, sr=sr, n_mels=128, fmax=8000, hop_length=256)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                              hop_length=256, ax=axes[0], cmap='inferno')
axes[0].set_title("Sheikh Al-Minshawi (Surah 1: Al-Fatiha) - Waqf & Takrar-Corrected Alignment", fontsize=14, fontweight='bold')
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

axes[1].set_ylim(0, 1)
axes[1].set_yticks([])
axes[1].set_title("Waqf-Corrected Letter Karaoke Stream", fontsize=12, fontweight='bold')
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
out_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_001_waqf_corrected.png"
plt.savefig(out_plot, dpi=200)
print(f"[Visualizer] Plot saved to: {out_plot}")
