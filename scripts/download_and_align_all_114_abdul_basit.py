#!/usr/bin/env python3
import urllib.request
import json
import os
import concurrent.futures
import time

BASE_URL = "https://everyayah.com/data/Abdul_Basit_Murattal_192kbps"
DATA_DIR = "/home/grem3/mahquranapp/public/data"
AUDIO_DIR = "/home/grem3/mahquranapp/public/audio/abdul_basit_murattal"
TIMING_DIR = os.path.join(DATA_DIR, "abdul_basit_murattal")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TIMING_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "verses_v4.json"), "r", encoding="utf-8") as f:
    ALL_VERSES = json.load(f)

BITRATES_V1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
SAMPLERATES_V1 = [44100, 48000, 32000, 0]

DIACRITICS = set([
    'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ',
    'ٓ', 'ٔ', 'ٕ', 'ٖ', 'ٗ', '٘', 'ٜ', 'ٝ',
    'ٞ', 'ٟ', 'ٰ', 'ۡ', '۟', '۠', 'ۢ', 'ۣ'
])

def split_arabic(text):
    chunks = []
    curr = ''
    for char in text:
        if char in DIACRITICS:
            curr += char
        else:
            if curr: chunks.append(curr)
            curr = char
    if curr: chunks.append(curr)
    return chunks

def analyze_audio_frames(raw_mp3):
    pos = 10 if raw_mp3.startswith(b'ID3') else 0
    if raw_mp3.startswith(b'ID3'):
        tag_size = ((raw_mp3[6] & 0x7F) << 21) | ((raw_mp3[7] & 0x7F) << 14) | ((raw_mp3[8] & 0x7F) << 7) | (raw_mp3[9] & 0x7F)
        pos = 10 + tag_size
        
    frames = []
    pure_bytes = []
    while pos < len(raw_mp3) - 4:
        b0, b1, b2, b3 = raw_mp3[pos], raw_mp3[pos+1], raw_mp3[pos+2], raw_mp3[pos+3]
        if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
            version = (b1 >> 3) & 0x03
            layer = (b1 >> 1) & 0x03
            bitrate_idx = (b2 >> 4) & 0x0F
            sr_idx = (b2 >> 2) & 0x03
            padding = (b2 >> 1) & 0x01
            if version == 3 and layer == 1 and bitrate_idx < 15 and sr_idx < 3:
                bitrate = BITRATES_V1_L3[bitrate_idx] * 1000
                sr = SAMPLERATES_V1[sr_idx]
                flen = (144 * bitrate) // sr + padding
                if flen <= 0 or pos + flen > len(raw_mp3):
                    pos += 1
                    continue
                fbytes = raw_mp3[pos:pos+flen]
                pure_bytes.append(fbytes)
                e = sum((b - 128)**2 for b in fbytes[6:]) / max(1, len(fbytes) - 6)
                t = len(frames) * (1152.0 / sr)
                frames.append((t, e))
                pos += flen
                continue
        pos += 1
        
    # VAD: Voice Onset & Offset
    max_e = max(e for _, e in frames) if frames else 1.0
    min_e = min(e for _, e in frames) if frames else 0.0
    threshold = min_e + (max_e - min_e) * 0.12
    
    v_start = 0.0
    for t, e in frames:
        if e > threshold:
            v_start = t
            break
            
    v_end = frames[-1][0] if frames else 0.0
    for t, e in reversed(frames):
        if e > threshold:
            v_end = t
            break
            
    total_dur = frames[-1][0] if frames else 0.0
    return b''.join(pure_bytes), total_dur, v_start, v_end, frames

def get_tajweed_phonetic_weight(chunk, is_word_end, is_ayah_end, base_char, ayah_num, surah_num):
    has_maddah = ('ٓ' in chunk) or ('ۤ' in chunk) or ('ٓ' in chunk)
    has_shaddah = ('ّ' in chunk) or ('ّ' in chunk)
    has_dagger_alif = ('ٰ' in chunk) or ('ٰ' in chunk)
    has_sukoon = ('ْ' in chunk) or ('ۡ' in chunk) or ('ْ' in chunk)

    # 1. Madd Lazim (6 Harakat)
    if has_maddah or ('ضَّآ' in chunk) or ('ضَّ' in chunk and surah_num == 1 and ayah_num == 7):
        return 7.5
    # 2. Madd Arid li-s-Sukoon at Ayah Stop (4-6 Harakat)
    if is_ayah_end and is_word_end and (base_char in 'وي' or has_dagger_alif or 'ي' in chunk or 'و' in chunk):
        return 5.0
    # 3. Ghunnah on Shaddah Nun/Mim
    if has_shaddah and base_char in 'نم':
        return 2.6
    # 4. Standard Shaddah
    if has_shaddah:
        return 1.9
    # 5. Natural Madd / Dagger Alif
    if has_dagger_alif or (base_char in 'اوي' and not has_sukoon and len(chunk) == 1):
        return 2.2
    # 6. Sukoon / Qalqalah
    if has_sukoon:
        return 0.75
    # 7. Hamzat Wasl / silent Lam in Al-
    if base_char == 'ٱ' or (base_char == 'ل' and not has_sukoon and not has_shaddah):
        return 0.45
    return 1.0

def fetch_ayah(surah_num, ayah_num):
    url = f"{BASE_URL}/{surah_num:03d}{ayah_num:03d}.mp3"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except Exception as e:
            if attempt == 3: raise e
            time.sleep(0.5)

def process_surah(surah_num):
    s_key = str(surah_num)
    verses = ALL_VERSES.get(s_key, [])
    if not verses:
        return f"Surah {surah_num}: Not found"

    out_audio_path = os.path.join(AUDIO_DIR, f"surah_{surah_num:03d}.mp3")
    out_timing_path = os.path.join(TIMING_DIR, f"letter_timing_{surah_num}.json")

    # Download ayahs in parallel
    ayah_nums = [v["ayah"] for v in verses]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        future_to_ayah = {pool.submit(fetch_ayah, surah_num, ay): ay for ay in ayah_nums}
        ayah_audio_map = {}
        for fut in concurrent.futures.as_completed(future_to_ayah):
            ay = future_to_ayah[fut]
            ayah_audio_map[ay] = fut.result()

    master_audio_bytes = []
    final_letters = []
    cur_global_time = 0.0
    global_word_idx = 0

    for v_idx, verse in enumerate(verses):
        ayah_num = verse["ayah"]
        raw_mp3 = ayah_audio_map[ayah_num]
        pure_bytes, total_dur, v_start, v_end, frames = analyze_audio_frames(raw_mp3)
        master_audio_bytes.append(pure_bytes)

        words = verse["words"]
        vocal_duration = max(0.5, v_end - v_start)

        # Word weights
        word_weights = []
        for w in words:
            chunks = split_arabic(w["arabic"])
            w_w = sum(get_tajweed_phonetic_weight(c, i == len(chunks) - 1, False, c[0], ayah_num, surah_num) for i, c in enumerate(chunks))
            word_weights.append(w_w)

        total_ayah_w = sum(word_weights)
        cur_w_start = cur_global_time + v_start

        for w_rel_idx, w in enumerate(words):
            is_ayah_end_word = (w_rel_idx == len(words) - 1)
            w_dur = (vocal_duration * word_weights[w_rel_idx]) / total_ayah_w
            w_end = cur_w_start + w_dur

            chunks = split_arabic(w["arabic"])
            letter_weights = [
                get_tajweed_phonetic_weight(c, i == len(chunks) - 1, is_ayah_end_word, c[0], ayah_num, surah_num)
                for i, c in enumerate(chunks)
            ]
            total_l_w = sum(letter_weights)

            cur_l_start = cur_w_start
            for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
                l_dur = (w_dur * lw) / total_l_w
                l_end = w_end if l_idx == len(chunks) - 1 else cur_l_start + l_dur

                final_letters.append({
                    "charIdx": len(final_letters),
                    "charIdxInWord": l_idx,
                    "char": chunk,
                    "start": round(cur_l_start, 3),
                    "end": round(l_end, 3),
                    "duration": round(l_end - cur_l_start, 3),
                    "wordIdx": global_word_idx,
                    "ayah": ayah_num,
                    "verseIdx": v_idx
                })
                cur_l_start = l_end

            cur_w_start = w_end
            global_word_idx += 1

        cur_global_time += total_dur

    with open(out_audio_path, "wb") as f:
        f.write(b"".join(master_audio_bytes))

    with open(out_timing_path, "w", encoding="utf-8") as f:
        json.dump(final_letters, f, ensure_ascii=False, indent=2)

    return f"Surah {surah_num:3d}: SawtTajweed Calibrated ({len(verses):3d} Ayahs, {len(final_letters):4d} letters)"

if __name__ == "__main__":
    print("Recalibrating full Quran with SawtTajweed Acoustic Precision Engine...")
    surahs = list(range(1, 115))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_surah, s): s for s in surahs}
        for fut in concurrent.futures.as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                print(res)
            except Exception as e:
                print(f"Surah {s} failed: {e}")
