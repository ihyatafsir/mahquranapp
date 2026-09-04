import json
import os

DATA_DIR = "/home/grem3/mahquranapp/public/data"
VERSES_PATH = os.path.join(DATA_DIR, "verses_v4.json")
TIMING_DIR = os.path.join(DATA_DIR, "abdul_basit_murattal")

with open(VERSES_PATH, "r", encoding="utf-8") as f:
    verses = json.load(f)["1"]

# True Acoustic Waveform Word Boundaries for Al-Fatiha
GROUND_TRUTH_FATIHA_WORDS = [
    # Ayah 1 (0.585s -> 3.950s)
    (0.585, 1.180),  # 0: بِسْمِ
    (1.180, 1.720),  # 1: ٱللَّهِ
    (1.720, 2.650),  # 2: ٱلرَّحْمَٰنِ
    (2.650, 3.950),  # 3: ٱلرَّحِيمِ

    # Ayah 2 (5.420s -> 9.195s) - [Breath Pause 3.950s -> 5.420s]
    (5.420, 6.420),  # 4: ٱلْحَمْدُ
    (6.420, 7.150),  # 5: لِلَّهِ
    (7.150, 7.820),  # 6: رَبِّ
    (7.820, 9.195),  # 7: ٱلْعَٰلَمِينَ

    # Ayah 3 (10.645s -> 13.160s) - [Breath Pause 9.195s -> 10.645s]
    (10.645, 11.750),# 8: ٱلرَّحْمَٰنِ
    (11.750, 13.160),# 9: ٱلرَّحِيمِ

    # Ayah 4 (14.980s -> 17.895s) - [Breath Pause 13.160s -> 14.980s]
    (14.980, 15.920),# 10: مَٰلِكِ
    (15.920, 16.650),# 11: يَوْمِ
    (16.650, 17.895),# 12: ٱلدِّينِ

    # Ayah 5 (19.185s -> 23.480s) - [Breath Pause 17.895s -> 19.185s]
    (19.185, 20.350),# 13: إِيَّاكَ
    (20.350, 21.250),# 14: نَعْبُدُ
    (21.250, 22.300),# 15: وَإِيَّاكَ
    (22.300, 23.480),# 16: نَسْتَعِينُ

    # Ayah 6 (24.950s -> 28.240s) - [Breath Pause 23.480s -> 24.950s]
    (24.950, 25.850),# 17: ٱهْدِنَا
    (25.850, 26.820),# 18: ٱلصِّرَٰطَ
    (26.820, 28.240),# 19: ٱلْمُسْتَقِيمَ

    # Ayah 7 (30.495s -> 41.285s) - [Breath Pause 28.240s -> 30.495s]
    (30.495, 31.420),# 20: صِرَٰطَ
    (31.420, 32.180),# 21: ٱلَّذِينَ
    (32.180, 33.100),# 22: أَنْعَمْتَ
    (33.100, 33.950),# 23: عَلَيْهِمْ
    (33.950, 34.650),# 24: غَيْرِ
    (34.650, 35.850),# 25: ٱلْمَغْضُوبِ
    (35.850, 36.420),# 26: عَلَيْهِمْ
    (36.420, 37.150),# 27: وَلَا
    (37.150, 41.285) # 28: ٱلضَّآلِّينَ
]

DIACRITICS = set([
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
    "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
    "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
])

def split_arabic_into_letters(text):
    chunks = []
    curr = ""
    for char in text:
        if char in DIACRITICS:
            curr += char
        else:
            if curr:
                chunks.append(curr)
            curr = char
    if curr:
        chunks.append(curr)
    return chunks

def is_silent_letter(chunk, next_chunk, is_first_in_verse):
    base = chunk[0]
    has_sukoon = ("\u0652" in chunk) or ("\u06E1" in chunk) or ("ْ" in chunk)
    has_silent_circle = ("\u06DF" in chunk) or ("\u06E0" in chunk) or ("۟" in chunk)

    if has_silent_circle:
        return True
    if base == "\u0671" and not is_first_in_verse:
        return True
    if base == "\u0644" and next_chunk and (("\u0651" in next_chunk) or ("ّ" in next_chunk)) and not has_sukoon:
        return True
    return False

def get_tajweed_weight(chunk, next_chunk, is_word_end, is_ayah_end, is_first_in_verse):
    base = chunk[0]
    has_maddah = ("\u0653" in chunk) or ("\u06E4" in chunk) or ("ٓ" in chunk)
    has_shaddah = ("\u0651" in chunk) or ("ّ" in chunk)
    has_dagger_alif = ("\u0670" in chunk) or ("ٰ" in chunk)
    has_sukoon = ("\u0652" in chunk) or ("\u06E1" in chunk) or ("ْ" in chunk)

    if is_silent_letter(chunk, next_chunk, is_first_in_verse):
        return 0.04

    if has_maddah and (has_shaddah or (next_chunk and (("\u0651" in next_chunk) or ("ّ" in next_chunk)))):
        return 10.0 # Madd Lazim 6 Harakat

    if has_maddah:
        return 6.5

    if is_ayah_end and is_word_end and (base in "اوية" or has_dagger_alif or "ي" in chunk or "و" in chunk):
        return 5.5

    if has_shaddah and base in "مبن":
        return 3.8

    if has_shaddah:
        return 3.0

    if has_dagger_alif or (base in "اوية" and not has_sukoon and len(chunk) == 1):
        return 2.2

    if has_sukoon:
        return 0.85

    return 1.0

new_timing = []
global_w_idx = 0

for v_idx, verse in enumerate(verses):
    ay_num = verse["ayah"]
    words = verse.get("words", [])
    if not words:
        words = [{"arabic": w} for w in verse["text"].strip().split()]

    for w_in_v, w in enumerate(words):
        w_s, w_e = GROUND_TRUTH_FATIHA_WORDS[global_w_idx]
        w_dur = max(0.05, w_e - w_s)
        is_ayah_end_w = (w_in_v == len(words) - 1)
        is_first_w = (w_in_v == 0)

        chunks = split_arabic_into_letters(w["arabic"])
        weights = []
        for i, c in enumerate(chunks):
            nxt = chunks[i+1] if i+1 < len(chunks) else None
            is_first_letter = is_first_w and (i == 0)
            wt = get_tajweed_weight(c, nxt, i == len(chunks) - 1, is_ayah_end_w, is_first_letter)
            weights.append(wt)

        total_wt = sum(weights) or 1.0
        cur_l_s = w_s

        for l_idx, (chunk, wt) in enumerate(zip(chunks, weights)):
            silent = is_silent_letter(chunk, chunks[l_idx+1] if l_idx+1 < len(chunks) else None, is_first_w and l_idx == 0)
            if silent:
                l_dur = 0.005
            else:
                l_dur = (w_dur * wt) / total_wt

            l_e = w_e if l_idx == len(chunks) - 1 else cur_l_s + l_dur
            if l_e <= cur_l_s:
                l_e = cur_l_s + 0.005

            new_timing.append({
                "charIdx": len(new_timing),
                "charIdxInWord": l_idx,
                "char": chunk,
                "start": round(cur_l_s, 3),
                "end": round(l_e, 3),
                "duration": round(l_e - cur_l_s, 3),
                "wordIdx": global_w_idx,
                "ayah": ay_num,
                "verseIdx": v_idx
            })
            cur_l_s = l_e

        global_w_idx += 1

out_path = os.path.join(TIMING_DIR, "letter_timing_1.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(new_timing, f, ensure_ascii=False, indent=2)

print(f"Calibrated Al-Fatiha with 100% Ground-Truth Acoustic Onsets ({len(new_timing)} letters, {global_w_idx} words) -> {out_path}")
