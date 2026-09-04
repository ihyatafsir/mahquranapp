#!/usr/bin/env python3
import urllib.request
import json
import os
import math

BASE_URL = "https://everyayah.com/data/Abdul_Basit_Murattal_192kbps"
DATA_DIR = "/home/grem3/mahquranapp/public/data"
AUDIO_DIR = "/home/grem3/mahquranapp/public/audio/abdul_basit_murattal"
TIMING_DIR = os.path.join(DATA_DIR, "abdul_basit_murattal")

with open(os.path.join(DATA_DIR, "verses_v4.json"), "r", encoding="utf-8") as f:
    ALL_VERSES = json.load(f)

DIACRITICS = set([
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
    "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
    "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
])

def split_arabic(text):
    chunks = []
    curr = ""
    for char in text:
        if char in DIACRITICS: curr += char
        else:
            if curr: chunks.append(curr)
            curr = char
    if curr: chunks.append(curr)
    return chunks

# True Neural Word Clamps from Whisper
NEURAL_WORD_CLAMPS = [
    # Ayah 1 (4.36s)
    (0.000, 1.020),  # 0: بِسْمِ
    (1.020, 1.520),  # 1: ٱللَّهِ
    (1.520, 2.500),  # 2: ٱلرَّحْمَٰنِ
    (2.500, 3.960),  # 3: ٱلرَّحِيمِ

    # Ayah 2 (5.30s)
    (4.360, 6.100),  # 4: ٱلْحَمْدُ
    (6.100, 6.920),  # 5: لِلَّهِ
    (6.920, 7.760),  # 6: رَبِّ
    (7.760, 9.600),  # 7: ٱلْعَٰلَمِينَ

    # Ayah 3 (4.02s)
    (9.660, 11.600), # 8: ٱلرَّحْمَٰنِ
    (11.600, 13.600),# 9: ٱلرَّحِيمِ

    # Ayah 4 (4.54s)
    (13.690, 15.750),# 10: مَٰلِكِ
    (15.750, 16.770),# 11: يَوْمِ
    (16.770, 18.200),# 12: ٱلدِّينِ

    # Ayah 5 (5.62s)
    (18.230, 19.850),# 13: إِيَّاكَ
    (19.850, 20.950),# 14: نَعْبُدُ
    (20.950, 22.100),# 15: وَإِيَّاكَ
    (22.100, 23.800),# 16: نَسْتَعِينُ

    # Ayah 6 (4.81s)
    (24.500, 25.730),# 17: ٱهْدِنَا
    (25.730, 26.650),# 18: ٱلصِّرَٰطَ
    (26.650, 28.500),# 19: ٱلْمُسْتَقِيمَ

    # Ayah 7 (13.14s)
    (29.800, 30.960),# 20: صِرَٰطَ
    (30.960, 31.840),# 21: ٱلَّذِينَ
    (31.840, 32.660),# 22: أَنْعَمْتَ
    (32.660, 33.640),# 23: عَلَيْهِمْ
    (33.640, 34.220),# 24: غَيْرِ
    (34.220, 35.460),# 25: ٱلْمَغْضُوبِ
    (35.460, 36.080),# 26: عَلَيْهِمْ
    (36.080, 36.900),# 27: وَلَا
    (36.900, 41.500) # 28: ٱلضَّآلِّينَ
]

# Physical Articulatory Profile Mapping (Lisan al-Arab & Mushafahah Physics)
def get_articulatory_profile(chunk, next_chunk, is_word_end, is_ayah_end, ayah_num):
    base = chunk[0]
    has_maddah = ("\u0653" in chunk) or ("\u06E4" in chunk) or ("ٓ" in chunk)
    has_shaddah = ("\u0651" in chunk) or ("ّ" in chunk)
    has_dagger_alif = ("\u0670" in chunk) or ("ٰ" in chunk)
    has_sukoon = ("\u0652" in chunk) or ("\u06E1" in chunk) or ("ْ" in chunk)

    # 1. Elided Silent Letters (Hamzat Wasl, Lam Shamsiyyah): 0ms duration / instantaneous handoff
    if base == "\u0671":
        return {"art": "silent_handoff", "weight": 0.05}
    if base == "\u0644" and next_chunk and (("\u0651" in next_chunk) or ("ّ" in next_chunk)) and not has_sukoon:
        return {"art": "silent_handoff", "weight": 0.05}

    # 2. Madd Lazim (6 Harakat in Dhaallin): Jawf Full Open Extension
    if has_maddah or ("ضَّ" in chunk and ayah_num == 7):
        return {"art": "jawf_madd_6", "weight": 9.5}

    # 3. Madd Arid li-s-Sukoon at Ayah pauses (ar-Raheem, al-Alameen, ad-Deen):
    if is_ayah_end and is_word_end and (base in "وي" or has_dagger_alif or "ي" in chunk or "و" in chunk):
        return {"art": "madd_arid_4", "weight": 5.8}

    # 4. Bilabial Lip Closure with Shaddah (e.g. Meem / Baa with Shaddah):
    if has_shaddah and base in "مب":
        return {"art": "bilabial_shaddah", "weight": 3.6}

    # 5. General Shaddah (Lingual / Dental Hold):
    if has_shaddah:
        return {"art": "shaddah_hold", "weight": 3.0}

    # 6. Natural Madd (2 Harakat):
    if has_dagger_alif or (base in "اوية" and not has_sukoon and len(chunk) == 1):
        return {"art": "madd_asli_2", "weight": 2.2}

    # 7. Sibilant & Fricative Mouth Airflow (Seen, Saad, Faa, Sheen):
    if base in "سصفشزذظث":
        return {"art": "fricative_airflow", "weight": 1.2}

    # 8. Sukoon Stop:
    if has_sukoon:
        return {"art": "sukoon_stop", "weight": 0.85}

    return {"art": "vocal_nucleus", "weight": 1.0}

all_letters = []
global_w_idx = 0

for v_idx, verse in enumerate(ALL_VERSES["1"]):
    ay_num = verse["ayah"]
    words = verse["words"]
    
    for w_in_v, w in enumerate(words):
        w_s, w_e = NEURAL_WORD_CLAMPS[global_w_idx]
        w_dur = w_e - w_s
        is_ayah_end_w = (w_in_v == len(words) - 1)
        
        chunks = split_arabic(w["arabic"])
        profiles = [
            get_articulatory_profile(c, chunks[i+1] if i+1 < len(chunks) else None, i == len(chunks) - 1, is_ayah_end_w, ay_num)
            for i, c in enumerate(chunks)
        ]
        
        total_w = sum(p["weight"] for p in profiles) or 1.0
        cur_l_s = w_s
        
        for l_idx, (chunk, prof) in enumerate(zip(chunks, profiles)):
            l_dur = (w_dur * prof["weight"]) / total_w
            l_e = w_e if l_idx == len(chunks) - 1 else cur_l_s + l_dur
            
            all_letters.append({
                "charIdx": len(all_letters),
                "charIdxInWord": l_idx,
                "char": chunk,
                "start": round(cur_l_s, 3),
                "end": round(l_e, 3),
                "duration": round(l_e - cur_l_s, 3),
                "articulatoryAction": prof["art"],
                "wordIdx": global_w_idx,
                "ayah": ay_num,
                "verseIdx": v_idx
            })
            cur_l_s = l_e
            
        global_w_idx += 1

out_file = "/home/grem3/mahquranapp/public/data/abdul_basit_murattal/letter_timing_1.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_letters, f, ensure_ascii=False, indent=2)

print("Generated Articulatory Acoustic + Neural Clamped letter timing for Al-Fatiha!")
