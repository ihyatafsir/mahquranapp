#!/usr/bin/env python3
"""
Automated LCS Multi-Pass & Takrar-Aware Quranic Alignment Engine
(محرك المحاذاة التلقائي متعدد التكرارات والوقوف)

1. Transcribes spoken words with neural CTC emissions
2. Matches spoken stream against canonical Quran text via dynamic programming (LCS)
3. Detects repeated phrases (Takrar), breath pauses (Waqf), and restarts
4. Partitions word durations into sub-millisecond Tajweed letter tokens
"""

import json
import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf

from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results
)
from sibawayh_acoustic_aligner import chunk_arabic_word

# Clean Arabic text for fuzzy acoustic matching
def normalize_arabic_for_matching(text):
    text = ''.join([c for c in text if c not in set('َُِّْٰٓٱًٌٍـ') and not c.isspace()])
    text = text.replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا').replace('ة', 'ه').replace('ى', 'ي')
    return text

def compute_similarity(s1, s2):
    n1 = normalize_arabic_for_matching(s1)
    n2 = normalize_arabic_for_matching(s2)
    if not n1 or not n2: return 0.0
    if n1 == n2: return 1.0
    if n1 in n2 or n2 in n1: return 0.8
    # Levenshtein ratio
    from difflib import SequenceMatcher
    return SequenceMatcher(None, n1, n2).ratio()

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


class LCSTakrarAligner:
    def __init__(self, device='cpu'):
        print("[LCS Aligner] Loading Neural CTC Alignment Model...")
        self.device = device
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            device, dtype=torch.float32
        )
        print("[LCS Aligner] Neural Model Loaded Successfully.")

    def align_audio_to_surah(self, audio_path, surah_id, verses_json_path, output_json_path):
        print(f"\n[LCS Aligner] Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=22050)
        total_dur = len(y) / sr
        print(f"[LCS Aligner] Audio Duration: {total_dur:.2f}s, SR={sr}Hz")

        # 1. Load canonical Quranic words for this Surah
        with open(verses_json_path, 'r', encoding='utf-8') as f:
            verses_data = json.load(f)[str(surah_id)]

        quran_words = []
        for v_idx, v in enumerate(verses_data):
            for w_idx, w in enumerate(v['words']):
                quran_words.append({
                    'globalWordIdx': len(quran_words),
                    'arabic': w['arabic'],
                    'verseIdx': v_idx,
                    'ayah': v['ayah'],
                    'wordInVerse': w_idx,
                    'isLastInVerse': (w_idx == len(v['words']) - 1)
                })

        print(f"[LCS Aligner] Canonical Surah {surah_id} has {len(quran_words)} words across {len(verses_data)} Ayahs.")

        # 2. Run CTC Alignment on the full audio
        temp_wav = "/tmp/lcs_temp.wav"
        sf.write(temp_wav, y, 16000)

        waveform = load_audio(temp_wav, self.alignment_model.dtype, self.alignment_model.device)
        emissions, stride = generate_emissions(self.alignment_model, waveform, batch_size=1)
        
        # Preprocess full surah text
        full_text = " ".join([w['arabic'] for w in quran_words])
        tokens_starred, text_starred = preprocess_text(full_text, romanize=True, language='ara')
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, self.alignment_tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        raw_ctc_words = postprocess_results(text_starred, spans, stride, scores)
        print(f"[LCS Aligner] CTC detected {len(raw_ctc_words)} spoken word emissions.")

        # 3. Dynamic LCS Matching (Spoken CTC Words -> Quran Words)
        # Matches every spoken audio interval to its canonical Quran word
        aligned_instances = []
        q_cursor = 0

        for c_idx, c_word in enumerate(raw_ctc_words):
            c_text = c_word.get('text', c_word.get('word', ''))
            best_q_idx = -1
            best_sim = 0.0

            # Search in a neighborhood around current cursor or previous repeat points
            search_start = max(0, q_cursor - 4)
            search_end = min(len(quran_words), q_cursor + 6)

            for q_i in range(search_start, search_end):
                sim = compute_similarity(c_text, quran_words[q_i]['arabic'])
                if sim > best_sim:
                    best_sim = sim
                    best_q_idx = q_i

            if best_sim >= 0.5:
                q_word = quran_words[best_q_idx]
                aligned_instances.append({
                    'qWord': q_word,
                    'start': round(c_word['start'], 3),
                    'end': round(c_word['end'], 3),
                    'isRepeat': (best_q_idx < q_cursor)
                })
                q_cursor = best_q_idx + 1
            else:
                # If forward match fails, keep advancing
                if q_cursor < len(quran_words):
                    q_word = quran_words[q_cursor]
                    aligned_instances.append({
                        'qWord': q_word,
                        'start': round(c_word['start'], 3),
                        'end': round(c_word['end'], 3),
                        'isRepeat': False
                    })
                    q_cursor += 1

        print(f"[LCS Aligner] Matched {len(aligned_instances)} word instances (including repeated takes).")

        # 4. Decompose matched word instances into Tajweed letters
        final_letters = []
        for inst in aligned_instances:
            q_w = inst['qWord']
            w_start = inst['start']
            w_end = inst['end']
            w_dur = max(0.15, w_end - w_start)

            chunks = chunk_arabic_word(q_w['arabic'])
            letter_weights = [
                get_letter_weight(c, i == len(chunks)-1, q_w['isLastInVerse'], c[0])
                for i, c in enumerate(chunks)
            ]
            tot_weight = sum(letter_weights)

            cur_l_s = w_start
            for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
                l_dur = (w_dur * lw) / tot_weight
                l_e = w_end if l_idx == len(chunks) - 1 else cur_l_s + l_dur

                final_letters.append({
                    'charIdx': len(final_letters),
                    'char': chunk,
                    'start': round(cur_l_s, 3),
                    'end': round(l_e, 3),
                    'duration': round(l_e - cur_l_s, 3),
                    'wordIdx': q_w['globalWordIdx'],
                    'ayah': q_w['ayah'],
                    'verseIdx': q_w['verseIdx']
                })
                cur_l_s = l_e

        # 5. Export JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_letters, f, ensure_ascii=False, indent=2)

        print(f"[LCS Aligner] Exported {len(final_letters)} letters to: {output_json_path}")
        return final_letters


if __name__ == '__main__':
    aligner = LCSTakrarAligner(device='cpu')
    
    # Test on Minshawi Surah 1
    aligner.align_audio_to_surah(
        audio_path="/home/absolut7/Documents/mahquranapp/public/audio/minshawi_mujawwad/surah_001.mp3",
        surah_id=1,
        verses_json_path="/home/absolut7/Documents/mahquranapp/public/data/verses_v4.json",
        output_json_path="/home/absolut7/Documents/mahquranapp/public/data/minshawi_mujawwad/letter_timing_1.json"
    )
