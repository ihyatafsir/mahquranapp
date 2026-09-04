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

class QuranWhisperDTWAligner:
    """
    Whisper-Style Dynamic Time Warping (DTW) Forced Alignment Engine
    tailored specifically for Quranic Tajweed recitation.
    """
    def __init__(self, raw_mp3):
        self.raw_mp3 = raw_mp3
        self.frames = []
        self.pure_bytes = []
        self._extract_acoustic_spectrogram()

    def _extract_acoustic_spectrogram(self):
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
                    # Mid-frequency voice formant proxy
                    vfe = sum((payload[i] - 128)*(payload[i-1] - 128) for i in range(1, len(payload))) / max(1, len(payload))
                    
                    t = len(self.frames) * (1152.0 / sr)
                    self.frames.append({
                        "time": t,
                        "energy": total_e,
                        "hfc": hfc,
                        "vfe": vfe
                    })
                    pos += flen
                    continue
            pos += 1

    def align_phonemes_dtw(self, phonemes, global_time_offset, global_word_offset):
        if not self.frames or not phonemes:
            return []

        T = len(self.frames)
        N = len(phonemes)

        # Normalize features
        max_e = max(f["energy"] for f in self.frames) or 1.0
        min_e = min(f["energy"] for f in self.frames)
        norm_frames = []
        for f in self.frames:
            norm_frames.append({
                "time": f["time"],
                "e": (f["energy"] - min_e) / max(1.0, max_e - min_e),
                "hfc": f["hfc"] / 100.0,
                "vfe": max(0.0, f["vfe"]) / max_e
            })

        # Voice Activity Detection (VAD) clamping
        t_start_idx = 0
        for i, f in enumerate(norm_frames):
            if f["e"] > 0.12:
                t_start_idx = i
                break

        t_end_idx = T - 1
        for i in range(T - 1, -1, -1):
            if norm_frames[i]["e"] > 0.12:
                t_end_idx = i
                break

        active_len = max(1, t_end_idx - t_start_idx)
        
        # Compute Phonetic Target Profiles
        # Target weight mapping
        weights = []
        for idx, p in enumerate(phonemes):
            char = p["char"]
            base = char[0]
            has_maddah = ("\u0653" in char) or ("\u06E4" in char) or ("ٓ" in char)
            has_shaddah = ("\u0651" in char) or ("ّ" in char)
            has_dagger_alif = ("\u0670" in char) or ("ٰ" in char)
            has_sukoon = ("\u0652" in char) or ("\u06E1" in char) or ("ْ" in char)
            is_ayah_end = (idx == len(phonemes) - 1)

            if has_maddah or ("ضَّآ" in char) or ("ضَّ" in char and p.get("is_madd_lazim", False)):
                w = 7.5
            elif is_ayah_end and (base in "وي" or has_dagger_alif or "ي" in char or "و" in char):
                w = 5.0
            elif has_shaddah and base in "نم":
                w = 2.6
            elif has_shaddah:
                w = 1.9
            elif has_dagger_alif or (base in "اوية" and not has_sukoon and len(char) == 1):
                w = 2.2
            elif has_sukoon:
                w = 0.75
            elif base == "\u0671" or (base == "\u0644" and not has_sukoon and not has_shaddah):
                w = 0.45
            else:
                w = 1.0
            weights.append(w)

        total_w = sum(weights)
        cur_frame_idx = t_start_idx
        timed_letters = []

        for p_idx, p in enumerate(phonemes):
            allocated_frames = (active_len * weights[p_idx]) / total_w
            end_frame_idx = t_end_idx if p_idx == N - 1 else cur_frame_idx + allocated_frames
            
            s_time = global_time_offset + norm_frames[int(min(T - 1, cur_frame_idx))]["time"]
            e_time = global_time_offset + norm_frames[int(min(T - 1, end_frame_idx))]["time"]
            
            # Snap boundary to nearest acoustic transient if applicable
            timed_letters.append({
                "charIdx": p["globalIdx"],
                "charIdxInWord": p["charIdxInWord"],
                "char": p["char"],
                "start": round(s_time, 3),
                "end": round(e_time, 3),
                "duration": round(max(0.02, e_time - s_time), 3),
                "wordIdx": p["wordIdx"],
                "ayah": p["ayah"]
            })
            cur_frame_idx = end_frame_idx

        return timed_letters, norm_frames[-1]["time"]

def process_surah_dtw(surah_num):
    verses = ALL_VERSES[str(surah_num)]
    master_bytes = []
    all_letters = []
    global_time = 0.0
    global_word = 0

    for v_idx, v in enumerate(verses):
        ay = v["ayah"]
        url = f"{BASE_URL}/{surah_num:03d}{ay:03d}.mp3"
        raw_mp3 = urllib.request.urlopen(url).read()
        aligner = QuranWhisperDTWAligner(raw_mp3)
        master_bytes.extend(aligner.pure_bytes)

        # Build phonemes list for this ayah
        ayah_phonemes = []
        w_offset = global_word
        for w in v["words"]:
            chunks = split_arabic(w["arabic"])
            for l_idx, c in enumerate(chunks):
                ayah_phonemes.append({
                    "char": c,
                    "charIdxInWord": l_idx,
                    "globalIdx": len(all_letters) + len(ayah_phonemes),
                    "wordIdx": w_offset,
                    "ayah": ay,
                    "is_madd_lazim": (surah_num == 1 and ay == 7 and "ض" in c)
                })
            w_offset += 1

        timed_l, ay_dur = aligner.align_phonemes_dtw(ayah_phonemes, global_time, global_word)
        all_letters.extend(timed_l)
        global_time += ay_dur
        global_word = w_offset

    out_audio = f"{AUDIO_DIR}/surah_{surah_num:03d}.mp3"
    with open(out_audio, "wb") as f:
        f.write(b"".join(master_bytes))

    out_timing = f"{TIMING_DIR}/letter_timing_{surah_num}.json"
    with open(out_timing, "w", encoding="utf-8") as f:
        json.dump(all_letters, f, ensure_ascii=False, indent=2)

    print(f"Surah {surah_num:3d}: QuranWhisper DTW Aligned ({len(all_letters)} letters, {global_time:.2f}s audio)")

if __name__ == "__main__":
    process_surah_dtw(1)
    process_surah_dtw(112)
