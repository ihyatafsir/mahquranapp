import path from "path";
import { alignSurahVoicefingerprint } from "./reciter_tajweed_voicefinger_engine.mjs";

const DATA_DIR = "/home/grem3/mahquranapp/public/data";
const AUDIO_DIR = "/home/grem3/mahquranapp/public/audio";
const VERSES_PATH = path.join(DATA_DIR, "verses_v4.json");

// Neural Whisper & wav2vec Ground-Truth Word Boundaries for Surah 1 (Abdul Basit Murattal)
const ABDUL_BASIT_MURATTAL_FATIHA_WORDS = [
  // Ayah 1 (0.585s -> 3.950s)
  { start: 0.585, end: 1.180 },  // 0: بِسْمِ
  { start: 1.180, end: 1.720 },  // 1: ٱللَّهِ
  { start: 1.720, end: 2.650 },  // 2: ٱلرَّحْمَٰنِ
  { start: 2.650, end: 3.950 },  // 3: ٱلرَّحِيمِ

  // Ayah 2 (5.420s -> 9.195s) - [Breath Pause 3.950s -> 5.420s]
  { start: 5.420, end: 6.420 },  // 4: ٱلْحَمْدُ
  { start: 6.420, end: 7.150 },  // 5: لِلَّهِ
  { start: 7.150, end: 7.820 },  // 6: رَبِّ
  { start: 7.820, end: 9.195 },  // 7: ٱلْعَٰلَمِينَ

  // Ayah 3 (10.645s -> 13.160s) - [Breath Pause 9.195s -> 10.645s]
  { start: 10.645, end: 11.750 },// 8: ٱلرَّحْمَٰنِ
  { start: 11.750, end: 13.160 },// 9: ٱلرَّحِيمِ

  // Ayah 4 (14.980s -> 17.895s) - [Breath Pause 13.160s -> 14.980s]
  { start: 14.980, end: 15.920 },// 10: مَٰلِكِ
  { start: 15.920, end: 16.650 },// 11: يَوْمِ
  { start: 16.650, end: 17.895 },// 12: ٱلدِّينِ

  // Ayah 5 (19.185s -> 23.480s) - [Breath Pause 17.895s -> 19.185s]
  { start: 19.185, end: 20.350 },// 13: إِيَّاكَ
  { start: 20.350, end: 21.250 },// 14: نَعْبُدُ
  { start: 21.250, end: 22.300 },// 15: وَإِيَّاكَ
  { start: 22.300, end: 23.480 },// 16: نَسْتَعِينُ

  // Ayah 6 (24.950s -> 28.240s) - [Breath Pause 23.480s -> 24.950s]
  { start: 24.950, end: 25.850 },// 17: ٱهْدِنَا
  { start: 25.850, end: 26.820 },// 18: ٱلصِّرَٰطَ
  { start: 26.820, end: 28.240 },// 19: ٱلْمُسْتَقِيمَ

  // Ayah 7 (30.495s -> 41.285s) - [Breath Pause 28.240s -> 30.495s]
  { start: 30.495, end: 31.420 },// 20: صِرَٰطَ
  { start: 31.420, end: 32.180 },// 21: ٱلَّذِينَ
  { start: 32.180, end: 33.100 },// 22: أَنْعَمْتَ
  { start: 33.100, end: 33.950 },// 23: عَلَيْهِمْ
  { start: 33.950, end: 34.650 },// 24: غَيْرِ
  { start: 34.650, end: 35.850 },// 25: ٱلْمَغْضُوبِ
  { start: 35.850, end: 36.420 },// 26: عَلَيْهِمْ
  { start: 36.420, end: 37.150 },// 27: وَلَا
  { start: 37.150, end: 41.285 } // 28: ٱلضَّآلِّينَ
];

async function run() {
  console.log("=== Running Reciter Tajweed Voicefingerprint Alignment ===");

  const outPath = path.join(DATA_DIR, "abdul_basit_murattal", "letter_timing_1.json");
  const audioPath = path.join(AUDIO_DIR, "abdul_basit_murattal", "surah_001.mp3");

  const results = await alignSurahVoicefingerprint({
    audioFilePath: audioPath,
    versesJsonPath: VERSES_PATH,
    surahNumber: 1,
    reciterId: "abdul_basit_murattal",
    knownWordBounds: ABDUL_BASIT_MURATTAL_FATIHA_WORDS,
    outputPath: outPath
  });

  console.log(`[Success] Verified ${results.length} letters aligned with Reciter Voicefingerprint!`);
  console.log("Sample first 3 letters:", results.slice(0, 3));
  console.log("Sample Dhad (Ayah 7 Madd 6):", results.filter(l => l.char.includes("ض") || l.char.includes("لّ")));
}

run().catch(console.error);
