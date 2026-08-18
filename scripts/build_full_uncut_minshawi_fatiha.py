#!/usr/bin/env python3
"""
100% Uncut Audio & Multi-Take Repetition Highlighting for Sheikh Al-Minshawi (Surah 1: Al-Fatiha)
- Audio is 100% UNTOUCHED, UNCUT original EveryAyah files (001001.mp3 to 001007.mp3, 130.95s total).
- Quran text is 100% UNTOUCHED canonical text.
- Highlighting dynamically repeats and rewinds when the reciter repeats phrases (Takrar / Wasl):
    1. Ayah 1 (4.40s - 11.70s): [بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ] (Take 1)
    2. Ayah 2 (17.73s - 24.14s): [ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ] (Take 1)
    3. Ayah 3 (30.34s - 42.74s): [ٱلرَّحْمَٰنِ ٱلرَّحِيمِ] (Take 1)
    4. Ayah 4 (44.38s - 70.32s):
         - 49.70s - 56.20s -> REPEATS Ayah 2 [ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ] (Take 2)
         - 56.20s - 61.85s -> REPEATS Ayah 3 [ٱلرَّحْمَٰنِ ٱلرَّحِيمِ] (Take 2)
         - 61.85s - 68.23s -> Recites Ayah 4 [مَٰلِكِ يَوْمِ ٱلدِّينِ] (Take 1)
    5. Ayah 5 (75.70s - 84.16s): [إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ] (Take 1)
    6. Ayah 6 (90.14s - 96.11s): [ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ] (Take 1)
    7. Ayah 7 (96.10s - 130.95s):
         - 96.15s - 109.61s -> Part A [صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ]
         - 109.61s - 118.58s -> Breath Pause
         - 118.58s - 130.95s -> Part B [غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ]
"""

import json
import os
import numpy as np
import librosa
import soundfile as sf

from sibawayh_acoustic_aligner import chunk_arabic_word

# Load canonical verse structures
with open("/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json", 'r', encoding='utf-8') as f:
    verses_data = json.load(f)["1"]

# Mapping of every physical recitation event across the 7 raw audio files
RAW_AUDIO_EVENTS = [
    # File 1: Ayah 1
    (1, [(1, 0, 4, 4.40, 11.70)]),
    # File 2: Ayah 2
    (2, [(2, 0, 4, 4.41, 10.82)]),
    # File 3: Ayah 3
    (3, [(3, 0, 2, 3.96, 16.36)]),
    # File 4: Reciter chants Ayah 2 + Ayah 3 + Ayah 4!
    (4, [
        (2, 0, 4, 5.32, 11.80),    # Repeated Ayah 2
        (3, 0, 2, 11.80, 17.50),   # Repeated Ayah 3
        (4, 0, 3, 17.50, 23.85)    # Ayah 4
    ]),
    # File 5: Ayah 5
    (5, [(5, 0, 4, 5.38, 13.84)]),
    # File 6: Ayah 6
    (6, [(6, 0, 3, 4.90, 10.87)]),
    # File 7: Ayah 7 split by Waqf
    (7, [
        (7, 0, 4, 0.05, 13.51),    # Part A: صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ
        (7, 4, 9, 22.48, 34.85)    # Part B: غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ
    ])
]

sr = 22050
raw_audio_clips = []
final_letters = []
cur_file_offset = 0.0

ayah_word_offsets = {}
w_count = 0
for v_idx, v in enumerate(verses_data):
    ayah_word_offsets[v_idx + 1] = w_count
    w_count += len(v['words'])

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

# Process all 7 raw EveryAyah files as-is
for file_num, events in RAW_AUDIO_EVENTS:
    raw_path = f"/tmp/minshawi_001/00100{file_num}.mp3"
    y_raw, _ = librosa.load(raw_path, sr=sr)
    raw_dur = len(y_raw) / sr

    print(f"\n==================== RAW FILE {file_num} (Offset: {cur_file_offset:.2f}s, Dur: {raw_dur:.2f}s) ====================")

    for target_ayah, w_s_idx, w_e_idx, v_s, v_e in events:
        target_verse_info = verses_data[target_ayah - 1]
        target_words = target_verse_info['words'][w_s_idx:w_e_idx]
        event_dur = v_e - v_s
        ayah_base_w_idx = ayah_word_offsets[target_ayah]

        event_word_weights = []
        for w in target_words:
            chunks = chunk_arabic_word(w['arabic'])
            w_w = sum(get_letter_weight(c, i == len(chunks)-1, False, c[0]) for i, c in enumerate(chunks))
            event_word_weights.append(w_w)

        tot_event_weight = sum(event_word_weights)
        cur_w_start = cur_file_offset + v_s

        for w_rel_idx, w in enumerate(target_words):
            w_abs_idx = w_s_idx + w_rel_idx
            global_w_idx = ayah_base_w_idx + w_abs_idx
            is_ayah_end_word = (w_abs_idx == len(target_verse_info['words']) - 1)
            is_seg_end_word = (w_rel_idx == len(target_words) - 1)

            w_dur = (event_dur * event_word_weights[w_rel_idx]) / tot_event_weight
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
                    'charIdxInWord': l_idx,
                    'char': chunk,
                    'start': round(cur_l_start, 3),
                    'end': round(l_e, 3),
                    'duration': round(l_e - cur_l_start, 3),
                    'wordIdx': global_w_idx,
                    'ayah': target_ayah,
                    'verseIdx': target_ayah - 1
                })
                cur_l_start = l_e

            print(f"  [Ayah {target_ayah} Word #{global_w_idx}] '{w['arabic']}': {cur_w_start:.3f}s - {w_end:.3f}s (dur: {w_dur:.3f}s)")
            cur_w_start = w_end

    raw_audio_clips.append(y_raw)
    cur_file_offset += raw_dur

y_master = np.concatenate(raw_audio_clips)
master_audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3"
sf.write(master_audio_path, y_master, sr)
print(f"\n[Master Audio] Exported 100% uncut audio: {master_audio_path} (Duration: {len(y_master)/sr:.2f}s)")

output_json = "/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_letters, f, ensure_ascii=False, indent=2)
print(f"[Timing JSON] Exported {len(final_letters)} letter instances to: {output_json}")
