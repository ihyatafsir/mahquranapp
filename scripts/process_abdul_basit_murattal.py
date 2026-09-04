#!/usr/bin/env python3
import urllib.request
import json
import os

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
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
    "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
    "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
])

def split_arabic(text):
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

def parse_mp3_frames(buf):
    pos = 0
    total_samples = 0
    sample_rate = 44100
    frames_bytes = []
    
    if buf.startswith(b"ID3"):
        tag_size = ((buf[6] & 0x7F) << 21) | ((buf[7] & 0x7F) << 14) | ((buf[8] & 0x7F) << 7) | (buf[9] & 0x7F)
        pos = 10 + tag_size
        
    while pos < len(buf) - 4:
        b0, b1, b2, b3 = buf[pos], buf[pos+1], buf[pos+2], buf[pos+3]
        if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
            version = (b1 >> 3) & 0x03
            layer = (b1 >> 1) & 0x03
            bitrate_idx = (b2 >> 4) & 0x0F
            sr_idx = (b2 >> 2) & 0x03
            padding = (b2 >> 1) & 0x01
            
            if version == 3 and layer == 1 and bitrate_idx < 15 and sr_idx < 3:
                bitrate = BITRATES_V1_L3[bitrate_idx] * 1000
                sample_rate = SAMPLERATES_V1[sr_idx]
                frame_len = (144 * bitrate) // sample_rate + padding
                if frame_len <= 0 or pos + frame_len > len(buf):
                    pos += 1
                    continue
                frames_bytes.append(buf[pos:pos+frame_len])
                total_samples += 1152
                pos += frame_len
                continue
        pos += 1
        
    duration = total_samples / sample_rate if sample_rate else 0.0
    return b"".join(frames_bytes), duration

def get_tajweed_weight(chunk, is_word_end, is_ayah_end, base_char):
    has_maddah = ("\u0653" in chunk) or ("\u06E4" in chunk) or ("ٓ" in chunk)
    has_shaddah = ("\u0651" in chunk) or ("ّ" in chunk)
    has_dagger_alif = ("\u0670" in chunk) or ("ٰ" in chunk)
    has_sukoon = ("\u0652" in chunk) or ("\u06E1" in chunk) or ("ْ" in chunk)

    if has_maddah:
        return 5.5
    if is_ayah_end and is_word_end and (base_char in "وي" or has_dagger_alif):
        return 4.0
    if has_shaddah and base_char in "\u0646\u0645":
        return 2.5
    if has_shaddah:
        return 1.8
    if has_dagger_alif or (base_char in "\u0627\u0648\u064a" and not has_sukoon and len(chunk) == 1):
        return 2.0
    if has_sukoon:
        return 0.8
    return 1.0

def process_surah(surah_num):
    s_key = str(surah_num)
    verses = ALL_VERSES.get(s_key, [])
    if not verses:
        print("Surah " + str(surah_num) + " not found in verses_v4.json")
        return

    print("\n======================================================")
    print("Processing Sheikh Abdul Basit (Murattal) - Surah " + str(surah_num) + " (" + str(len(verses)) + " Ayahs)")
    print("======================================================")

    master_audio_bytes = []
    final_letters = []
    current_time_offset = 0.0
    global_word_idx = 0

    for v_idx, verse in enumerate(verses):
        ayah_num = verse["ayah"]
        url = BASE_URL + "/" + str(surah_num).zfill(3) + str(ayah_num).zfill(3) + ".mp3"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_mp3 = response.read()
        except Exception as e:
            print("Error downloading " + url + ": " + str(e))
            return

        pure_audio_bytes, ayah_dur = parse_mp3_frames(raw_mp3)
        master_audio_bytes.append(pure_audio_bytes)

        words = verse["words"]
        word_weights = []
        for w in words:
            chunks = split_arabic(w["arabic"])
            w_w = sum(get_tajweed_weight(c, i == len(chunks) - 1, False, c[0]) for i, c in enumerate(chunks))
            word_weights.append(w_w)

        total_ayah_weight = sum(word_weights)
        cur_w_start = current_time_offset

        for w_rel_idx, w in enumerate(words):
            is_ayah_end_word = (w_rel_idx == len(words) - 1)
            w_dur = (ayah_dur * word_weights[w_rel_idx]) / total_ayah_weight
            w_end = cur_w_start + w_dur

            chunks = split_arabic(w["arabic"])
            letter_weights = [
                get_tajweed_weight(c, i == len(chunks) - 1, is_ayah_end_word, c[0])
                for i, c in enumerate(chunks)
            ]
            total_l_weight = sum(letter_weights)

            cur_l_start = cur_w_start
            for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
                l_dur = (w_dur * lw) / total_l_weight
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

        current_time_offset += ayah_dur

    out_audio_path = os.path.join(AUDIO_DIR, "surah_" + str(surah_num).zfill(3) + ".mp3")
    with open(out_audio_path, "wb") as f:
        f.write(b"".join(master_audio_bytes))

    out_timing_path = os.path.join(TIMING_DIR, "letter_timing_" + str(surah_num) + ".json")
    with open(out_timing_path, "w", encoding="utf-8") as f:
        json.dump(final_letters, f, ensure_ascii=False, indent=2)

    print("  -> Exported Master Audio (" + str(round(current_time_offset, 2)) + "s) and " + str(len(final_letters)) + " letters timing to " + out_timing_path)

if __name__ == "__main__":
    surahs_to_process = [1, 108, 109, 110, 111, 112, 113, 114]
    for s in surahs_to_process:
        process_surah(s)
