#!/usr/bin/env python3
"""
High-Precision CTC + Visual-Acoustic Alignment for Sheikh Al-Minshawi (Surah 1: Al-Fatiha)
"""

import json
import os
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results
)
from sibawayh_acoustic_aligner import chunk_arabic_word

# Load verses from verses_v4.json
with open("/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json", 'r', encoding='utf-8') as f:
    v_list = json.load(f)["1"]

# Load CTC alignment model
device = 'cpu'
alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=torch.float32)

sr = 22050
ayah_clips = []
final_letters = []
global_word_idx = 0
cur_surah_offset = 0.0

for ayah_num in range(1, 8):
    audio_path = f"/tmp/minshawi_001/00100{ayah_num}.mp3"
    y_ayah, _ = librosa.load(audio_path, sr=sr)
    ayah_dur = len(y_ayah) / sr
    
    verse_data = v_list[ayah_num - 1]
    words_data = verse_data['words']
    verse_text = verse_data['text']
    
    # Run CTC forced alignment on this Ayah
    temp_wav = f"/tmp/minshawi_001/temp_ayah_{ayah_num}.wav"
    sf.write(temp_wav, y_ayah, 16000)
    
    waveform = load_audio(temp_wav, alignment_model.dtype, alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, waveform, batch_size=1)
    tokens_starred, text_starred = preprocess_text(verse_text, romanize=True, language='ara')
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank_token)
    ctc_words = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"\n==================== AYAH {ayah_num} (Offset: {cur_surah_offset:.2f}s, Dur: {ayah_dur:.2f}s) ====================")
    print(f"Verse Text: {verse_text}")
    print(f"Words: {len(words_data)} | CTC Matched Words: {len(ctc_words)}")
    
    # Map CTC words to verse words
    for w_idx, q_word in enumerate(words_data):
        if w_idx < len(ctc_words):
            c_w = ctc_words[w_idx]
            rel_start = c_w['start']
            rel_end = c_w['end']
        elif len(ctc_words) > 0:
            rel_start = ctc_words[-1]['end']
            rel_end = ayah_dur
        else:
            rel_start = 0.0
            rel_end = ayah_dur
            
        abs_start = cur_surah_offset + rel_start
        abs_end = cur_surah_offset + rel_end
        w_dur = max(0.15, abs_end - abs_start)
        
        # Letter chunking with Uthmani diacritics
        chunks = chunk_arabic_word(q_word['arabic'])
        
        # Tajweed duration weights
        weights = []
        is_last_word = (w_idx == len(words_data) - 1)
        for c_idx, c in enumerate(chunks):
            is_last_letter = (c_idx == len(chunks) - 1)
            if 'ٓ' in c:
                w = 5.5  # Madd Lazim
            elif 'ّ' in c:
                w = 2.8 if any(x in c for x in 'نم') else 2.0
            elif 'ٰ' in c or (c in 'اوي' and len(c) == 1):
                w = 2.2  # Madd Tabii'i
            elif is_last_word and is_last_letter and any(x in c for x in 'قطبجد'):
                w = 3.5  # Waqf Qalqalah
            elif is_last_word and is_last_letter and (c in 'وينمرل' or 'ْ' in c):
                w = 3.0  # Madd 'Arid li-Sukoon
            elif 'ْ' in c:
                w = 0.8
            else:
                w = 1.0
            weights.append(w)
            
        tot_w = sum(weights)
        cur_l_start = abs_start
        for c_idx, (chunk, w) in enumerate(zip(chunks, weights)):
            c_dur = (w_dur * w) / tot_w
            c_end = abs_end if c_idx == len(chunks) - 1 else cur_l_start + c_dur
            
            final_letters.append({
                'charIdx': len(final_letters),
                'char': chunk,
                'start': round(cur_l_start, 3),
                'end': round(c_end, 3),
                'duration': round(c_end - cur_l_start, 3),
                'wordIdx': global_word_idx,
                'ayah': ayah_num,
                'verseIdx': ayah_num - 1
            })
            cur_l_start = c_end
            
        print(f"  Word #{global_word_idx} '{q_word['arabic']}': {abs_start:.3f}s - {abs_end:.3f}s (dur: {w_dur:.3f}s, letters: {len(chunks)})")
        global_word_idx += 1
        
    ayah_clips.append(y_ayah)
    cur_surah_offset += ayah_dur

# Concatenate master audio for Surah 1
y_full = np.concatenate(ayah_clips)
master_audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3"
sf.write(master_audio_path, y_full, sr)
print(f"\n[Master Audio] Saved continuous Surah 1 audio to: {master_audio_path} (Duration: {len(y_full)/sr:.2f}s)")

output_json = "/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_letters, f, ensure_ascii=False, indent=2)
print(f"[Timing JSON] Saved {len(final_letters)} letters to: {output_json}")

# Generate High-Resolution Visual Spectrogram
total_dur = len(y_full) / sr
time_axis = np.linspace(0, total_dur, len(y_full))

fig, axes = plt.subplots(3, 1, figsize=(28, 12), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.3]})

S = librosa.feature.melspectrogram(y=y_full, sr=sr, n_mels=128, fmax=8000, hop_length=256)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                              hop_length=256, ax=axes[0], cmap='inferno')
axes[0].set_title("Sheikh Al-Minshawi (Mujawwad Style) - Surah Al-Fatiha Mel-Spectrogram (7 Ayahs)", fontsize=14, fontweight='bold')
fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

axes[1].plot(time_axis, y_full, color='#00ff88', alpha=0.6, label='Audio Waveform')
axes[1].set_title("Acoustic Amplitude & Energy Envelopes", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.2)

axes[2].set_ylim(0, 1)
axes[2].set_yticks([])
axes[2].set_title("Continuous Letter Karaoke Highlighting with Complete Diacritics", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12', '#064e3b']

for l in final_letters:
    dur = l['end'] - l['start']
    if dur <= 0: continue
    col = colors[l['wordIdx'] % len(colors)]
    rect = plt.Rectangle((l['start'], 0.1), dur, 0.8, color=col, alpha=0.85, ec='#38bdf8', lw=1.2)
    axes[2].add_patch(rect)
    mid_x = (l['start'] + l['end']) / 2
    axes[2].text(mid_x, 0.5, l['char'], fontsize=9, color='#ffffff', ha='center', va='center', fontweight='bold', fontname='DejaVu Sans')
    for ax in axes:
        ax.axvline(x=l['start'], color='#38bdf8', linestyle='--', alpha=0.3, lw=0.5)

plt.tight_layout()
out_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_001_spectrogram.png"
plt.savefig(out_plot, dpi=200)
print(f"[Visualizer] Saved spectrogram plot to: {out_plot}")
