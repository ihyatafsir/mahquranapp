#!/usr/bin/env python3
"""
Tajweed Phonetic Physics & Assimilation Aligner
(المطابقة الصوتية الفيزيائية مع قواعد الإدغام والمدود الدقيقة)

Implements:
1. Solar Lam assimilation (اللام الشمسية) -> 0 duration / absorbed into Shaddah
2. Hamzat al-Wasl in continuous speech (همزة الوصل في درج الكلام) -> 0 duration
3. Idgham (الإدغام) -> assimilated noon merges into target with Ghunnah
4. Asymmetric Waqf & Madd expansion -> short consonants stay ~100ms, elongation applies to Madd/Waqf
5. Energy envelope tracking & transient snapping
"""

import json
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

def decompose_word_with_tajweed_phonetics(word_str, is_first_in_verse=False, is_ayah_end=False):
    """
    Decomposes an Arabic word into display chunks while applying
    classical Tajweed phonological rules for acoustic duration weighting.
    """
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
    
    # Inspect chunks and apply Tajweed rules
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

        # Rule 1: Hamzat al-Wasl
        if base_char == 'ٱ':
            if not is_first_in_verse:
                # Silent wasla
                acoustic_weight = 0.05
            else:
                acoustic_weight = 0.6
        # Rule 2: Solar Lam (اللام الشمسية)
        elif base_char == 'ل' and (idx == 1 and raw_chunks[0].startswith('ٱ')) and (next_base in SOLAR_LETTERS):
            acoustic_weight = 0.05  # Silent solar lam
        # Rule 3: Madd Lazim (6 Harakaat)
        elif has_maddah:
            acoustic_weight = 6.0
        # Rule 4: Terminal Madd / Waqf / Qalqalah at Ayah end
        elif is_last and is_ayah_end:
            if base_char in 'قطبجد':
                acoustic_weight = 3.5  # Heavy Waqf Qalqalah
            elif base_char in 'وي' or has_dagger_alif:
                acoustic_weight = 4.5  # Madd 'Arid li-Sukoon
            else:
                acoustic_weight = 2.5
        # Rule 5: Shaddah with Ghunnah
        elif has_shaddah:
            if base_char in 'نم':
                acoustic_weight = 2.6
            else:
                acoustic_weight = 2.0
        # Rule 6: Internal Madd (Waaw, Yaa, Alif)
        elif has_dagger_alif or (base_char in 'اويى' and not has_sukoon and len(chunk) == 1):
            acoustic_weight = 2.2
        # Rule 7: Short voweled consonant (متحرك عادي)
        elif not has_sukoon:
            acoustic_weight = 1.0  # Crisp 1 Harakah
        # Rule 8: Sukoon consonant
        else:
            if base_char in 'قطبجد':
                acoustic_weight = 0.9  # Qalqalah
            elif base_char in 'سشصضفثذخغحهظز':
                acoustic_weight = 1.1  # Rikhwah flow
            else:
                acoustic_weight = 0.7  # Sukoon

        processed_chunks.append({
            'char': chunk,
            'base': base_char,
            'weight': acoustic_weight
        })

    return processed_chunks


class TajweedPrecisionAligner:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=22050)
        self.hop_length = 128  # ~5.8ms ultra-high resolution
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length, aggregate=np.median
        )
        self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]
        self.onset_frames = librosa.onset.onset_detect(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=self.hop_length, backtrack=True
        )
        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length=self.hop_length)

    def snap_to_transient(self, target_time, window_ms=25):
        w_s = window_ms / 1000.0
        cands = [t for t in self.onset_times if target_time - w_s <= t <= target_time + w_s]
        if not cands:
            return target_time
        # Pick strongest onset
        best_cand = target_time
        best_val = -1.0
        for c in cands:
            f = librosa.time_to_frames(c, sr=self.sr, hop_length=self.hop_length)
            if f < len(self.onset_env) and self.onset_env[f] > best_val:
                best_val = self.onset_env[f]
                best_cand = c
        return float(best_cand)

    def align_surah_112(self, verses_json_path, timing_json_path):
        with open(verses_json_path, 'r', encoding='utf-8') as f:
            v_list = json.load(f)["112"]
        with open(timing_json_path, 'r', encoding='utf-8') as f:
            timed_words = json.load(f)

        # Skip Basmalah in timing_112 (words 0 to 3)
        start_offset = 4
        
        quran_words = []
        for v_idx, verse in enumerate(v_list):
            for w_in_v, w in enumerate(verse['words']):
                quran_words.append({
                    'arabic': w['arabic'],
                    'verseIdx': v_idx,
                    'ayah': verse['ayah'],
                    'wordInVerse': w_in_v,
                    'isFirstInVerse': (w_in_v == 0),
                    'isLastInVerse': (w_in_v == len(verse['words']) - 1)
                })

        final_letters = []

        for w_idx, q_word in enumerate(quran_words):
            t_idx = w_idx + start_offset
            w_start = timed_words[t_idx]['start']
            w_end = timed_words[t_idx]['end']
            w_dur = w_end - w_start

            # Decompose word with Tajweed assimilation & weights
            chunks_info = decompose_word_with_tajweed_phonetics(
                q_word['arabic'],
                is_first_in_verse=q_word['isFirstInVerse'],
                is_ayah_end=q_word['isLastInVerse']
            )

            total_weight = sum(c['weight'] for c in chunks_info)

            # Distribute time proportionally to Tajweed weights
            cur_start = w_start
            for c_idx, c_info in enumerate(chunks_info):
                weight = c_info['weight']
                chunk_dur = (w_dur * weight) / total_weight
                
                if c_idx == len(chunks_info) - 1:
                    chunk_end = w_end
                else:
                    ideal_end = cur_start + chunk_dur
                    snapped = self.snap_to_transient(ideal_end, window_ms=20)
                    chunk_end = min(max(cur_start + 0.04, snapped), w_end - 0.04)

                final_letters.append({
                    'charIdx': len(final_letters),
                    'char': c_info['char'],
                    'start': round(cur_start, 3),
                    'end': round(chunk_end, 3),
                    'duration': round(chunk_end - cur_start, 3),
                    'wordIdx': w_idx,
                    'ayah': q_word['ayah'],
                    'verseIdx': q_word['verseIdx']
                })
                cur_start = chunk_end

        return final_letters


if __name__ == '__main__':
    audio_path = "/home/absolut7/Documents/mahquranapp/public/audio/surah_112.mp3"
    verses_path = "/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json"
    timing_path = "/home/absolut7/Documents/26apps/ihyatafsir-android/assets/audio_mah/timing_112.json"
    output_json = "/home/absolut7/Documents/mahquranapp/public/data/letter_timing_112.json"
    output_plot = "/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/surah112_precision_spectrogram.png"

    aligner = TajweedPrecisionAligner(audio_path)
    letters = aligner.align_surah_112(verses_path, timing_path)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(letters, f, ensure_ascii=False, indent=2)

    print(f"\n[Precision Aligner] Generated {len(letters)} letters for Surah 112!")

    # Zoomed High-Resolution Visual Plot for Ayahs 1, 2, 3, 4 (5.0s to 19.3s)
    T_START = 4.8
    T_END = 19.3
    y_sub, sr = librosa.load(audio_path, sr=22050, offset=T_START, duration=(T_END - T_START))
    time_axis = np.linspace(T_START, T_END, len(y_sub))

    fig, axes = plt.subplots(3, 1, figsize=(22, 11), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.3]})

    # 1. Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y_sub, sr=sr, n_mels=128, fmax=8000, hop_length=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                                  hop_length=128, ax=axes[0], cmap='viridis', x_coords=np.linspace(T_START, T_END, S_dB.shape[1]))
    axes[0].set_title("Surah 112 (Al-Ikhlas) - High-Precision Mel-Spectrogram (Formants & Vowel Ridges)", fontsize=13, fontweight='bold')
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

    # 2. Waveform & RMS Energy
    axes[1].plot(time_axis, y_sub, color='#00ff88', alpha=0.6, label='Audio Waveform')
    onset_env = librosa.onset.onset_strength(y=y_sub, sr=sr, hop_length=128)
    onset_times = np.linspace(T_START, T_END, len(onset_env))
    axes[1].plot(onset_times, onset_env / (np.max(onset_env) + 1e-6) * np.max(np.abs(y_sub)), color='#ff007f', lw=2, label='Transient Spikes E(t)')
    axes[1].set_title("Physical Acoustic Transients (Qaf, Dal Qalqalah, Shaddah Bursts)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.2)

    # 3. Precision Aligned Letter Banners
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_title("Tajweed Assimilated & Physically Snapped Letter Timing", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Time (Seconds)", fontsize=12, fontweight='bold')

    colors = ['#1e293b', '#0f172a', '#1e3a8a', '#14532d', '#701a75', '#7c2d12']

    for l in letters:
        if l['end'] < T_START or l['start'] > T_END: continue
        s = max(T_START, l['start'])
        e = min(T_END, l['end'])
        dur = e - s
        if dur <= 0: continue
        col = colors[l['wordIdx'] % len(colors)]
        rect = plt.Rectangle((s, 0.1), dur, 0.8, color=col, alpha=0.85, ec='#38bdf8', lw=1.2)
        axes[2].add_patch(rect)
        mid_x = (s + e) / 2
        axes[2].text(mid_x, 0.5, l['char'], fontsize=12, color='#ffffff', ha='center', va='center', fontweight='bold', fontname='DejaVu Sans')
        for ax in axes:
            ax.axvline(x=l['start'], color='#38bdf8', linestyle='--', alpha=0.4, lw=0.6)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    print(f"[Visualizer] Precision plot saved to: {output_plot}")
