import fs from "fs";
import { MPEGDecoder } from "mpg123-decoder";
import { pipeline } from "@xenova/transformers";

async function alignAllAyahs() {
  console.log("Loading Whisper ONNX Neural Model...");
  const transcriber = await pipeline("automatic-speech-recognition", "Xenova/whisper-tiny", {
    quantized: true,
  });
  
  const mp3Buffer = fs.readFileSync("/home/grem3/mahquranapp/public/audio/abdul_basit_murattal/surah_001.mp3");
  const decoder = new MPEGDecoder();
  await decoder.ready;
  const uint8 = new Uint8Array(mp3Buffer.buffer, mp3Buffer.byteOffset, mp3Buffer.byteLength);
  const { channelData, sampleRate } = decoder.decode(uint8);
  
  const orig = channelData[0];
  const targetSr = 16000;
  const targetLen = Math.floor(orig.length * targetSr / sampleRate);
  const pcm16k = new Float32Array(targetLen);
  for (let i = 0; i < targetLen; i++) {
    const srcIdx = (i * sampleRate) / targetSr;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(orig.length - 1, i0 + 1);
    const frac = srcIdx - i0;
    pcm16k[i] = orig[i0] * (1 - frac) + orig[i1] * frac;
  }
  
  // Ayah slices in seconds:
  const ayahWindows = [
    { ay: 1, s: 0.0, e: 4.36 },
    { ay: 2, s: 4.36, e: 9.66 },
    { ay: 3, s: 9.66, e: 13.69 },
    { ay: 4, s: 13.69, e: 18.23 },
    { ay: 5, s: 18.23, e: 23.85 },
    { ay: 6, s: 23.85, e: 28.66 },
    { ay: 7, s: 28.66, e: 41.80 }
  ];
  
  const allNeuralWords = [];
  for (const win of ayahWindows) {
    const startSample = Math.floor(win.s * 16000);
    const endSample = Math.floor(win.e * 16000);
    const slice = pcm16k.subarray(startSample, endSample);
    
    const res = await transcriber(slice, {
      language: "arabic",
      task: "transcribe",
      return_timestamps: "word",
    });
    
    console.log(`=== Ayah ${win.ay} Neural Result (${win.s}s -> ${win.e}s) ===`);
    if (res.chunks) {
      for (const ch of res.chunks) {
        const globalStart = Number((win.s + (ch.timestamp[0] || 0)).toFixed(3));
        const globalEnd = Number((win.s + (ch.timestamp[1] || 0)).toFixed(3));
        console.log(`  "${ch.text.trim()}" -> [${globalStart}s - ${globalEnd}s]`);
        allNeuralWords.push({
          ayah: win.ay,
          text: ch.text.trim(),
          start: globalStart,
          end: globalEnd
        });
      }
    }
  }
  
  fs.writeFileSync("/home/grem3/mahquranapp/public/data/abdul_basit_murattal/whisper_neural_fatiha.json", JSON.stringify(allNeuralWords, null, 2));
  console.log("Saved neural word bounds to whisper_neural_fatiha.json");
}

alignAllAyahs().catch(console.error);
