#!/usr/bin/env python3
import urllib.request
import json
import os
import math
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
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
    "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
    "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
])

def split_into_letters(text):
    chunks = []
    curr = ""
    for char in text:
        if char in DIACRITICS:
            curr += char
        else:
            if curr: chunks.append(curr)
            curr = char
    if curr: chunks.append(curr)
    return chunks

class LisanWaveEngine:
    def __init__(self, raw_mp3):
        self.raw_mp3 = raw_mp3
        self.frames = []
        self.pure_bytes = []
        self.v_start = 0.0
        self.v_end = 0.0
        self.total_dur = 0.0
        self._analyze_audio()

    def _analyze_audio(self):
        buf = self.raw_mp3
        pos = 10 if buf.startswith(b"ID3") else 0
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
                    sr = SAMPLERATES_V1[sr_idx]
                    flen = (144 * bitrate) // sr + padding
                    if flen <= 0 or pos + flen > len(buf):
                        pos += 1
                        continue
                    fbytes = buf[pos:pos+flen]
                    self.pure_bytes.append(fbytes)
                    payload = fbytes[6:]
                    energy = sum((b - 128)**2 for b in payload) / max(1, len(payload))
                    flux = sum(abs(payload[i] - payload[i-1]) for i in range(1, len(payload))) / max(1, len(payload))
                    t = len(self.frames) * (1152.0 / sr)
                    self.frames.append({"time": t, "energy": energy, "flux": flux})
                    pos += flen
                    continue
            pos += 1

        if not self.frames:
            return

        self.total_dur = self.frames[-1]["time"]
        max_e = max(f["energy"] for f in self.frames) or 1.0
        min_e = min(f["energy"] for f in self.frames)
        threshold = min_e + (max_e - min_e) * 0.12

        for f in self.frames:
            if f["energy"] > threshold:
                self.v_start = f["time"]
                break

        for f in reversed(self.frames):
            if f["energy"] > threshold:
                self.v_end = f["time"]
                break

def get_lisan_makhraj_weight(chunk, is_word_end, is_ayah_end, ayah_num, surah_num):
    has_maddah = ("\u0653" in chunk) or ("\u06E4" in chunk) or ("ٓ" in chunk)
    has_shaddah = ("\u0651" in chunk) or ("ّ" in chunk)
    has_dagger_alif = ("\u0670" in chunk) or ("ٰ" in chunk)
    has_sukoon = ("\u0652" in chunk) or ("\u06E1" in chunk) or ("ْ" in chunk)
    base = chunk[0]

    if has_maddah or ("ضَّ" in chunk and surah_num == 1 and ayah_num == 7):
        return 7.0
    if is_ayah_end and is_word_end and (base in "وي" or has_dagger_alif or "ي" in chunk or "و" in chunk):
        return 4.8
    if has_shaddah and base in "نم":
        return 2.5
    if has_shaddah:
        return 1.8
    if has_dagger_alif or (base in "اوية" and not has_sukoon and len(chunk) == 1):
        return 2.0
    if has_sukoon:
        return 0.8
    if base == "\u0671" or (base == "\u0644" and not has_sukoon and not has_shaddah):
        return 0.45
    return 1.0

def align_lisan_verse(engine, verse, surah_num, ayah_num, global_time_offset, global_word_offset):
    words = verse["words"]
    vocal_dur = max(0.5, engine.v_end - engine.v_start)
    w_start = global_time_offset + engine.v_start

    word_weights = []
    for w_in_v, w in enumerate(words):
        chunks = split_into_letters(w["arabic"])
        is_ayah_end_w = (w_in_v == len(words) - 1)
        w_w = sum(get_lisan_makhraj_weight(c, i == len(chunks) - 1, is_ayah_end_w, ayah_num, surah_num) for i, c in enumerate(chunks))
        word_weights.append(w_w)

    total_w_weight = sum(word_weights)
    timed_letters = []
    cur_w_start = w_start
    w_idx = global_word_offset

    for w_in_v, w in enumerate(words):
        is_ayah_end_w = (w_in_v == len(words) - 1)
        w_dur = (vocal_dur * word_weights[w_in_v]) / total_w_weight
        w_end = cur_w_start + w_dur

        chunks = split_into_letters(w["arabic"])
        letter_weights = [
            get_lisan_makhraj_weight(c, i == len(chunks) - 1, is_ayah_end_w, ayah_num, surah_num)
            for i, c in enumerate(chunks)
        ]
        total_l_w = sum(letter_weights)
        cur_l_start = cur_w_start

        for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
            l_dur = (w_dur * lw) / total_l_w
            l_end = w_end if l_idx == len(chunks) - 1 else cur_l_start + l_dur

            timed_letters.append({
                "charIdx": len(timed_letters),
                "charIdxInWord": l_idx,
                "char": chunk,
                "start": round(cur_l_start, 3),
                "end": round(l_end, 3),
                "duration": round(l_end - cur_l_start, 3),
                "wordIdx": w_idx,
                "ayah": ayah_num
            })
            cur_l_start = l_end

        cur_w_start = w_end
        w_idx += 1

    return timed_letters, engine.total_dur, w_idx

def process_surah(surah_num):
    verses = ALL_VERSES.get(str(surah_num), [])
    if not verses: return

    out_audio = os.path.join(AUDIO_DIR, f"surah_{surah_num:03d}.mp3")
    out_timing = os.path.join(TIMING_DIR, f"letter_timing_{surah_num}.json")

    ayah_nums = [v["ayah"] for v in verses]
    def fetch_ayah(ay):
        url = f"{BASE_URL}/{surah_num:03d}{ay:03d}.mp3"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return ay, resp.read()
            except Exception:
                time.sleep(0.4)
        return ay, b""

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        ayah_audio_map = dict(pool.map(fetch_ayah, ayah_nums))

    master_bytes = []
    all_letters = []
    global_time = 0.0
    global_word = 0

    for v in verses:
        ay = v["ayah"]
        raw_mp3 = ayah_audio_map[ay]
        engine = LisanWaveEngine(raw_mp3)
        master_bytes.extend(engine.pure_bytes)
        letters, ay_dur, global_word = align_lisan_verse(
            engine, v, surah_num, ay, global_time, global_word
        )
        for l in letters:
            l["charIdx"] = len(all_letters)
            all_letters.append(l)
        global_time += ay_dur

    with open(out_audio, "wb") as f:
        f.write(b"".join(master_bytes))

    with open(out_timing, "w", encoding="utf-8") as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)

    return f"Surah {surah_num:3d}: Lisan Wave Aligned ({len(all_letters)} letters, {global_time:.1f}s)"

if __name__ == "__main__":
    print("=== Lisan al-Arab Wave Alignment Pipeline Running ===")
    surahs = list(range(1, 115))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_surah, s): s for s in surahs}
        for fut in concurrent.futures.as_completed(futures):
            s = futures[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"Surah {s} error: {e}")
