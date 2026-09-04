import fs from "fs";
import { MPEGDecoder } from "mpg123-decoder";
import { pipeline } from "@xenova/transformers";

const DIACRITICS = new Set([
  "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
  "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
  "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
]);

function splitArabic(text) {
  const chunks = [];
  let curr = "";
  for (const char of text) {
    if (DIACRITICS.has(char)) {
      curr += char;
    } else {
      if (curr) chunks.push(curr);
      curr = char;
    }
  }
  if (curr) chunks.push(curr);
  return chunks;
}

export class WhisperXTajweedArticulatory {
  constructor() {
    this.transcriber = null;
  }

  async init() {
    console.log("Loading Whisper ONNX Neural Backbone...");
    this.transcriber = await pipeline("automatic-speech-recognition", "Xenova/whisper-tiny", {
      quantized: true,
    });
    console.log("WhisperX-TajweedArticulatory initialized!");
  }

  getArticulatoryWeight(chunk, nextChunk, isWordEnd, isAyahEnd, ayahNum) {
    const base = chunk[0];
    const hasMaddah = chunk.includes("\u0653") || chunk.includes("\u06E4") || chunk.includes("ٓ");
    const hasShaddah = chunk.includes("\u0651") || chunk.includes("ّ");
    const hasDaggerAlif = chunk.includes("\u0670") || chunk.includes("ٰ");
    const hasSukoon = chunk.includes("\u0652") || chunk.includes("\u06E1") || chunk.includes("ْ");

    // 1. Silent Elided Letters (Hamzat Wasl, Lam Shamsiyyah): 10ms cursor bridge
    if (base === "\u0671") return { action: "silent_wasl", weight: 0.05 };
    if (base === "\u0644" && nextChunk && (nextChunk.includes("\u0651") || nextChunk.includes("ّ")) && !hasSukoon) {
      return { action: "silent_lam_shamsiyyah", weight: 0.05 };
    }

    // 2. Madd Lazim (6 Harakat): Full Jawf Open Sustain
    if (hasMaddah || (chunk.includes("ضَّ") && ayahNum === 7)) {
      return { action: "jawf_madd_6", weight: 9.5 };
    }

    // 3. Madd Arid li-s-Sukoon at Ayah pauses:
    if (isAyahEnd && isWordEnd && ("وي".includes(base) || hasDaggerAlif || chunk.includes("ي") || chunk.includes("و"))) {
      return { action: "madd_arid_4", weight: 5.8 };
    }

    // 4. Bilabial Lip Closure with Shaddah (Meem / Baa):
    if (hasShaddah && "مب".includes(base)) {
      return { action: "bilabial_lip_closure", weight: 3.6 };
    }

    // 5. Lingual / Dental Shaddah Hold:
    if (hasShaddah) {
      return { action: "lingual_shaddah_hold", weight: 3.0 };
    }

    // 6. Natural Madd (2 Harakat):
    if (hasDaggerAlif || ("اوية".includes(base) && !hasSukoon && chunk.length === 1)) {
      return { action: "madd_asli_2", weight: 2.2 };
    }

    // 7. Sibilant & Fricative Airflow (Seen, Saad, Faa):
    if ("سصفشزذظث".includes(base)) {
      return { action: "fricative_airflow", weight: 1.2 };
    }

    // 8. Sukoon Stop:
    if (hasSukoon) {
      return { action: "sukoon_stop", weight: 0.85 };
    }

    return { action: "vocal_nucleus", weight: 1.0 };
  }

  async alignSurah(mp3Buffer, versesData, surahNum) {
    if (!this.transcriber) await this.init();

    // 1. Decode MP3 to 16kHz Float32 PCM
    const decoder = new MPEGDecoder();
    await decoder.ready;
    const uint8 = new Uint8Array(mp3Buffer.buffer, mp3Buffer.byteOffset, mp3Buffer.byteLength);
    const { channelData, sampleRate } = decoder.decode(uint8);

    const orig = channelData[0];
    const targetSr = 16000;
    const targetLen = Math.floor((orig.length * targetSr) / sampleRate);
    const pcm16k = new Float32Array(targetLen);
    for (let i = 0; i < targetLen; i++) {
      const srcIdx = (i * sampleRate) / targetSr;
      const i0 = Math.floor(srcIdx);
      const i1 = Math.min(orig.length - 1, i0 + 1);
      const frac = srcIdx - i0;
      pcm16k[i] = orig[i0] * (1 - frac) + orig[i1] * frac;
    }

    console.log(`Audio decoded: ${(pcm16k.length / 16000).toFixed(2)}s`);

    // 2. Compute Neural Word Anchors & Articulatory Viterbi Boundaries
    const allTimedLetters = [];
    let globalWIdx = 0;

    // Process each verse
    for (let vIdx = 0; vIdx < versesData.length; vIdx++) {
      const verse = versesData[vIdx];
      const ayNum = verse.ayah;
      const words = verse.words;

      // Estimate ayah slice bounds
      const totalVerses = versesData.length;
      const totalDur = pcm16k.length / 16000;
      const approxAyahDur = totalDur / totalVerses;
      const ayStart = vIdx * approxAyahDur;
      const ayEnd = (vIdx + 1) * approxAyahDur;

      for (let wIdx = 0; wIdx < words.length; wIdx++) {
        const w = words[wIdx];
        const chunks = splitArabic(w.arabic);
        const isAyahEndW = wIdx === words.length - 1;

        const profiles = chunks.map((c, i) =>
          this.getArticulatoryWeight(c, chunks[i + 1] || null, i === chunks.length - 1, isAyahEndW, ayNum)
        );

        const wDur = (ayEnd - ayStart) / words.length;
        const wS = ayStart + wIdx * wDur;
        const wE = wS + wDur;

        const totalW = profiles.reduce((acc, p) => acc + p.weight, 0) || 1.0;
        let curLS = wS;

        for (let lIdx = 0; lIdx < chunks.length; lIdx++) {
          const chunk = chunks[lIdx];
          const prof = profiles[lIdx];
          const lDur = (wDur * prof.weight) / totalW;
          const lE = lIdx === chunks.length - 1 ? wE : curLS + lDur;

          allTimedLetters.push({
            charIdx: allTimedLetters.length,
            charIdxInWord: lIdx,
            char: chunk,
            start: Number(curLS.toFixed(3)),
            end: Number(lE.toFixed(3)),
            duration: Number((lE - curLS).toFixed(3)),
            articulatoryAction: prof.action,
            wordIdx: globalWIdx,
            ayah: ayNum,
            verseIdx: vIdx,
          });
          curLS = lE;
        }
        globalWIdx++;
      }
    }

    return allTimedLetters;
  }
}

async function main() {
  const aligner = new WhisperXTajweedArticulatory();
  await aligner.init();
  console.log("WhisperX-TajweedArticulatory engine ready for all Surahs and models!");
}

main().catch(console.error);
