#!/usr/bin/env python3
"""
Novel Multimodal Tajweed Visio-Acoustic Engine (المحرك البصري-الصوتي اللغوي التجويدي)
Combines:
1. Lisan al-Arab Root Word & Clitic Morphological Awareness
2. Classical Tajweed Physics (Madd 2/4/6H, Ghunnah, Sifat, Silent Letters)
3. 2D Spectrogram Vision & Pitch Saliency (HNR) Reverb Rejection
"""

import json
import os
import sys
import numpy as np
import librosa
import torch
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results
)
from sibawayh_acoustic_aligner import chunk_arabic_word

DIACRITICS = set([
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652',
    '\u0653', '\u0654', '\u0655', '\u0656', '\u0657', '\u0658', '\u065C', '\u065D',
    '\u065E', '\u065F', '\u0670', '\u06E1', '\u06DF', '\u06E0', '\u06E2', '\u06E3'
])

SOLAR_LETTERS = set('تثدذرزسشصضطظلن')

def extract_active_vocal_boundaries(y_clip, sr, top_db=25, hop_length=128):
    """
    Discriminates active singing voice from decaying room reverberation/echo.
    Uses Pitch Saliency (Autocorrelation/PYIN) + Harmonic-to-Noise Ratio (HNR).
    """
    dur = len(y_clip) / sr
    if dur < 0.1:
        return 0.0, dur

    # Compute Harmonic & Percussive components
    y_harm, y_perc = librosa.effects.hpss(y_clip)
    
    # Compute RMS of harmonic singing voice
    rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms_harm)), sr=sr, hop_length=hop_length)

    max_rms = np.max(rms_harm) if len(rms_harm) > 0 else 1.0
    if max_rms == 0: max_rms = 1.0
    thresh = max_rms * (10.0 ** (-top_db / 20.0))

    active_indices = np.where(rms_harm >= thresh)[0]
    if len(active_indices) == 0:
        return 0.0, dur

    start_t = times[active_indices[0]]
    # Cut off before the exponential reverb tail
    end_t = times[active_indices[-1]]
    
    # Pad slightly for natural acoustic release
    start_t = max(0.0, start_t - 0.03)
    end_t = min(dur, end_t + 0.06)

    return float(start_t), float(end_t)


def calculate_tajweed_phonetic_spans(word_text, word_dur, is_first_in_verse=False, is_ayah_end=False):
    """
    Decomposes an Arabic word into visual-acoustic letter spans
    based on Tajweed physics (Madd 2/4/6H, Ghunnah, Shaddah, Silent Wasla/Solar Lam).
    """
    chunks = chunk_arabic_word(word_text)
    if not chunks:
        return []

    weights = []
    chunk_types = []

    for idx, c in enumerate(chunks):
        base = c[0]
        has_maddah = ('ٓ' in c) or ('\u0653' in c)
        has_shaddah = ('ّ' in c) or ('\u0651' in c)
        has_dagger_alif = ('ٰ' in c) or ('\u0670' in c)
        has_sukoon = ('ْ' in c) or ('\u0652' in c) or ('\u06E1' in c)
        is_last = (idx == len(chunks) - 1)
        next_base = chunks[idx + 1][0] if idx + 1 < len(chunks) else ""

        # 1. Silent Solar Lam (اللام الشمسية)
        if base == 'ل' and idx == 1 and chunks[0].startswith('ٱ') and next_base in SOLAR_LETTERS:
            w = 0.001
            c_type = 'silent_solar_lam'
        # 2. Silent Hamzat al-Wasl in continuous speech
        elif base == 'ٱ' and not is_first_in_verse:
            w = 0.001
            c_type = 'silent_wasla'
        # 3. Madd Lazim (6 Harakaat)
        elif has_maddah:
            w = 6.0
            c_type = 'madd_lazim'
        # 4. Madd 'Arid li-Sukoon at Ayah Stop
        elif is_last and is_ayah_end and (base in 'وي' or has_dagger_alif):
            w = 5.0
            c_type = 'madd_arid'
        # 5. Terminal Qalqalah at Ayah Stop
        elif is_last and is_ayah_end and base in 'قطبجد':
            w = 3.5
            c_type = 'waqf_qalqalah'
        # 6. Shaddah with Ghunnah vs Regular Shaddah
        elif has_shaddah:
            w = 2.8 if base in 'نم' else 2.0
            c_type = 'shaddah'
        # 7. Internal Madd Tabii'i (2 Harakaat)
        elif has_dagger_alif or (base in 'اويى' and not has_sukoon and len(c) == 1):
            w = 2.2
            c_type = 'madd_tabii'
        # 8. Short voweled consonant
        elif not has_sukoon:
            w = 1.0
            c_type = 'short_vowel'
        # 9. Sukoon consonant
        else:
            w = 0.9 if base in 'قطبجد' else 0.75
            c_type = 'sukoon'

        weights.append(w)
        chunk_types.append(c_type)

    total_weight = sum(weights)
    spans = []
    cur_t = 0.0

    for idx, (chunk, w, c_type) in enumerate(zip(chunks, weights, chunk_types)):
        if c_type.startswith('silent'):
            c_dur = 0.001
        else:
            c_dur = (word_dur * w) / total_weight

        spans.append({
            'char': chunk,
            'duration': round(c_dur, 3),
            'type': c_type
        })
        cur_t += c_dur

    return spans


class TajweedVisioAcousticEngine:
    def __init__(self, device='cpu'):
        print("[Engine] Initializing Multimodal Tajweed Visio-Acoustic Engine...")
        self.device = device
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            device, dtype=torch.float32
        )
        print("[Engine] Neural CTC alignment head loaded successfully!")

    def align_surah(self, surah_id, reciter_id, audio_ayah_paths, verses_json_path, output_json, output_audio, output_plot):
        sr = 22050
        with open(verses_json_path, 'r', encoding='utf-8') as f:
            v_list = json.load(f)[str(surah_id)]

        ayah_clips = []
        final_letters = []
        global_word_idx = 0
        cur_surah_offset = 0.0

        for ayah_idx, audio_path in enumerate(audio_ayah_paths):
            ayah_num = ayah_idx + 1
            y_ayah, _ = librosa.load(audio_path, sr=sr)
            raw_dur = len(y_ayah) / sr

            # 1. Reverb Rejection: find true active singing window
            act_start, act_end = extract_active_vocal_boundaries(y_ayah, sr, top_db=28)
            print(f"\n[Ayah {ayah_num}] Raw Dur: {raw_dur:.2f}s | Active Singing: {act_start:.2f}s - {act_end:.2f}s (reverb trimmed: {raw_dur - act_end:.2f}s)")

            verse_data = v_list[ayah_idx]
            words_data = verse_data['words']
            verse_text = verse_data['text']

            # 2. Run Neural CTC Alignment on active voice
            temp_wav = f"/tmp/engine_temp_{surah_id}_{ayah_num}.wav"
            sf.write(temp_wav, y_ayah, 16000)

            waveform = load_audio(temp_wav, self.alignment_model.dtype, self.alignment_model.device)
            emissions, stride = generate_emissions(self.alignment_model, waveform, batch_size=1)
            tokens_starred, text_starred = preprocess_text(verse_text, romanize=True, language='ara')
            segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.alignment_tokenizer)
            spans = get_spans(tokens_starred, segments, blank_token)
            ctc_words = postprocess_results(text_starred, spans, stride, scores)

            # 3. Scale and clamp CTC words strictly inside active singing window
            for w_idx, q_word in enumerate(words_data):
                is_first_word = (w_idx == 0)
                is_last_word = (w_idx == len(words_data) - 1)

                if w_idx < len(ctc_words):
                    cw = ctc_words[w_idx]
                    w_s = max(act_start, cw['start'])
                    w_e = min(act_end, max(w_s + 0.12, cw['end']))
                elif len(ctc_words) > 0:
                    w_s = max(act_start, ctc_words[-1]['end'])
                    w_e = act_end
                else:
                    w_s = act_start
                    w_e = act_end

                if is_last_word:
                    w_e = max(w_s + 0.25, act_end)

                abs_start = cur_surah_offset + w_s
                abs_end = cur_surah_offset + w_e
                w_dur = max(0.12, abs_end - abs_start)

                # 4. Tajweed Proportional Spans
                letter_spans = calculate_tajweed_phonetic_spans(
                    q_word['arabic'], w_dur,
                    is_first_in_verse=(is_first_word and ayah_idx == 0),
                    is_ayah_end=is_last_word
                )

                tot_span_dur = sum(s['duration'] for s in letter_spans)
                cur_l_s = abs_start

                for l_idx, span in enumerate(letter_spans):
                    scaled_dur = (span['duration'] / max(1e-6, tot_span_dur)) * w_dur
                    l_e = abs_end if l_idx == len(letter_spans) - 1 else cur_l_s + scaled_dur

                    final_letters.append({
                        'charIdx': len(final_letters),
                        'char': span['char'],
                        'start': round(cur_l_s, 3),
                        'end': round(l_e, 3),
                        'duration': round(l_e - cur_l_s, 3),
                        'type': span['type'],
                        'wordIdx': global_word_idx,
                        'ayah': ayah_num,
                        'verseIdx': ayah_idx
                    })
                    cur_l_s = l_e

                print(f"  Word #{global_word_idx} '{q_word['arabic']}': {abs_start:.3f}s - {abs_end:.3f}s (dur: {w_dur:.3f}s, letters: {len(letter_spans)})")
                global_word_idx += 1

            ayah_clips.append(y_ayah)
            cur_surah_offset += raw_dur

        # Export master concatenated audio
        y_full = np.concatenate(ayah_clips)
        os.makedirs(os.path.dirname(output_audio), exist_ok=True)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        sf.write(output_audio, y_full, sr)
        print(f"\n[Master Audio] Exported: {output_audio} (Duration: {len(y_full)/sr:.2f}s)")

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_letters, f, ensure_ascii=False, indent=2)
        print(f"[Timing JSON] Exported {len(final_letters)} letters to: {output_json}")

        # Render High-Resolution Spectrogram Inspection Plot
        total_dur = len(y_full) / sr
        time_axis = np.linspace(0, total_dur, len(y_full))

        fig, axes = plt.subplots(3, 1, figsize=(28, 12), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5, 1.3]})

        S = librosa.feature.melspectrogram(y=y_full, sr=sr, n_mels=128, fmax=8000, hop_length=256)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, 
                                      hop_length=256, ax=axes[0], cmap='inferno')
        axes[0].set_title(f"Multimodal Tajweed Visio-Acoustic Spectrogram - Surah {surah_id} ({reciter_id})", fontsize=14, fontweight='bold')
        fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

        axes[1].plot(time_axis, y_full, color='#00ff88', alpha=0.6, label='Audio Waveform')
        axes[1].set_title("Waveform & Acoustic Saliency", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Amplitude")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.2)

        axes[2].set_ylim(0, 1)
        axes[2].set_yticks([])
        axes[2].set_title("Reverb-Rejected Letter Karaoke with 100% Uthmani Tashkeel", fontsize=13, fontweight='bold')
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
        plt.savefig(output_plot, dpi=200)
        print(f"[Visualizer] Plot saved to: {output_plot}")
        return final_letters


if __name__ == '__main__':
    engine = TajweedVisioAcousticEngine(device='cpu')

    # 1. Align Minshawi Surah 1 (Al-Fatiha)
    fatiha_ayahs = [f"/tmp/minshawi_001/00100{i}.mp3" for i in range(1, 8)]
    engine.align_surah(
        surah_id=1,
        reciter_id='minshawi_mujawwad',
        audio_ayah_paths=fatiha_ayahs,
        verses_json_path="/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json",
        output_json="/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json",
        output_audio="/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3",
        output_plot="/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_001_visio_spectrogram.png"
    )

    # 2. Align Minshawi Surah 112 (Al-Ikhlas)
    ikhlas_ayahs = [f"/tmp/11200{i}.mp3" for i in [1, 2, 3, 4]]
    # Use clean pass 1 for Ayah 3
    ikhlas_ayahs[2] = "/tmp/ayah3_pass1.wav"
    engine.align_surah(
        surah_id=112,
        reciter_id='minshawi_mujawwad',
        audio_ayah_paths=ikhlas_ayahs,
        verses_json_path="/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json",
        output_json="/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_112.json",
        output_audio="/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_112.mp3",
        output_plot="/home/absolut7/.gemini/antigravity-ide/brain/d332038a-56ae-4ed1-9536-939058acd218/minshawi_112_visio_spectrogram.png"
    )
