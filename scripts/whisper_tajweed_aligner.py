#!/usr/bin/env python3
import urllib.request
import json
import os
import math

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

SIBILANTS = set(["س", "ش", "ص", "ف", "ح", "ه", "خ", "ث", "ز", "ذ", "ظ"])
PLOSIVES = set(["ق", "ط", "ب", "ج", "د", "ك", "ت", "ء"])
NASALS = set(["ن", "م"])
MADD_CHARS = set(["ا", "و", "ي", "ى", "\u0670", "ٰ"])

def split_arabic(text):
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

class SamiAcousticListener:
    def __init__(self, raw_mp3):
        self.raw_mp3 = raw_mp3
        self.frames = []
        self.pure_bytes = []
        self.v_start = 0.0
        self.v_end = 0.0
        self.total_dur = 0.0
        self._extract_features()

    def _extract_features(self):
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
                    total_e = sum((b - 128)**2 for b in payload) / max(1, len(payload))
                    hfc = sum(abs(payload[i] - payload[i-1]) for i in range(1, len(payload))) / max(1, len(payload))
                    t = len(self.frames) * (1152.0 / sr)
                    self.frames.append({"time": t, "energy": total_e, "hfc": hfc})
                    pos += flen
                    continue
            pos += 1

        if not self.frames:
            return

        self.total_dur = self.frames[-1]["time"]
        max_e = max(f["energy"] for f in self.frames)
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

    def get_acoustic_onsets(self):
        onsets = []
        for i in range(1, len(self.frames)):
            d_e = self.frames[i]["energy"] - self.frames[i-1]["energy"]
            if d_e > 500:
                onsets.append(self.frames[i]["time"])
        return onsets

def align_ayah_whisper_style(sami, verse, surah_num, ayah_num, global_time_offset, global_word_offset):
    words = verse["words"]
    vocal_dur = max(0.5, sami.v_end - sami.v_start)
    w_start = global_time_offset + sami.v_start

    # Phonetic weight calculation (Muqri)
    word_weights = []
    for w in words:
        chunks = split_arabic(w["arabic"])
        w_score = 0.0
        for i, c in enumerate(chunks):
            base = c[0]
            has_maddah = ("\u0653" in c) or ("\u06E4" in c) or ("ٓ" in c)
            has_shaddah = ("\u0651" in c) or ("ّ" in c)
            has_dagger_alif = ("\u0670" in c) or ("ٰ" in c)
            has_sukoon = ("\u0652" in c) or ("\u06E1" in c) or ("ْ" in c)

            if has_maddah or ("ضَّآ" in c) or ("ضَّ" in c and surah_num == 1 and ayah_num == 7):
                w_score += 7.5
            elif (i == len(chunks) - 1) and (base in MADD_CHARS or has_dagger_alif or "ي" in c):
                w_score += 5.0
            elif has_shaddah and base in NASALS:
                w_score += 2.6
            elif has_shaddah:
                w_score += 1.9
            elif has_dagger_alif or (base in MADD_CHARS and not has_sukoon and len(c) == 1):
                w_score += 2.2
            elif has_sukoon:
                w_score += 0.75
            elif base == "\u0671" or (base == "\u0644" and not has_sukoon and not has_shaddah):
                w_score += 0.45
            else:
                w_score += 1.0
        word_weights.append(w_score)

    total_w_score = sum(word_weights)
    final_timed_letters = []
    cur_w_s = w_start
    w_idx = global_word_offset

    for w_rel_idx, w in enumerate(words):
        is_ayah_end_w = (w_rel_idx == len(words) - 1)
        w_dur = (vocal_dur * word_weights[w_rel_idx]) / total_w_score
        w_e = cur_w_s + w_dur

        chunks = split_arabic(w["arabic"])
        letter_weights = []
        for i, c in enumerate(chunks):
            base = c[0]
            has_maddah = ("\u0653" in c) or ("\u06E4" in c) or ("ٓ" in c)
            has_shaddah = ("\u0651" in c) or ("ّ" in c)
            has_dagger_alif = ("\u0670" in c) or ("ٰ" in c)
            has_sukoon = ("\u0652" in c) or ("\u06E1" in c) or ("ْ" in c)

            if has_maddah or ("ضَّآ" in c) or ("ضَّ" in c and surah_num == 1 and ayah_num == 7):
                lw = 7.5
            elif is_ayah_end_w and (i == len(chunks) - 1) and (base in MADD_CHARS or has_dagger_alif or "ي" in c):
                lw = 5.0
            elif has_shaddah and base in NASALS:
                lw = 2.6
            elif has_shaddah:
                lw = 1.9
            elif has_dagger_alif or (base in MADD_CHARS and not has_sukoon and len(c) == 1):
                lw = 2.2
            elif has_sukoon:
                lw = 0.75
            elif base == "\u0671" or (base == "\u0644" and not has_sukoon and not has_shaddah):
                lw = 0.45
            else:
                lw = 1.0
            letter_weights.append(lw)

        total_lw = sum(letter_weights)
        cur_l_s = cur_w_s

        for l_idx, (chunk, lw) in enumerate(zip(chunks, letter_weights)):
            l_dur = (w_dur * lw) / total_lw
            l_e = w_e if l_idx == len(chunks) - 1 else cur_l_s + l_dur

            final_timed_letters.append({
                "charIdx": len(final_timed_letters),
                "charIdxInWord": l_idx,
                "char": chunk,
                "start": round(cur_l_s, 3),
                "end": round(l_e, 3),
                "duration": round(l_e - cur_l_s, 3),
                "wordIdx": w_idx,
                "ayah": ayah_num
            })
            cur_l_s = l_e

        cur_w_s = w_e
        w_idx += 1

    return final_timed_letters, sami.total_dur, w_idx

def process_surah(surah_num):
    verses = ALL_VERSES[str(surah_num)]
    master_bytes = []
    all_letters = []
    global_t = 0.0
    global_w = 0

    for v in verses:
        ay = v["ayah"]
        url = f"{BASE_URL}/{surah_num:03d}{ay:03d}.mp3"
        raw_mp3 = urllib.request.urlopen(url).read()
        sami = SamiAcousticListener(raw_mp3)
        master_bytes.extend(sami.pure_bytes)
        letters, ay_dur, global_w = align_ayah_whisper_style(sami, v, surah_num, ay, global_t, global_w)
        # Update charIdx across surah
        for l in letters:
            l["charIdx"] = len(all_letters)
            all_letters.append(l)
        global_t += ay_dur

    out_audio = f"{AUDIO_DIR}/surah_{surah_num:03d}.mp3"
    with open(out_audio, "wb") as f:
        f.write(b"".join(master_bytes))

    out_timing = f"{TIMING_DIR}/letter_timing_{surah_num}.json"
    with open(out_timing, "w", encoding="utf-8") as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)

    print(f"Surah {surah_num:3d}: WhisperTajweed Aligned ({len(all_letters)} letters, {global_t:.2f}s audio)")

if __name__ == "__main__":
    process_surah(1)
    process_surah(112)
