import fs from "fs";
import path from "path";
import { MPEGDecoder } from "mpg123-decoder";

const AUDIO_PATH = "/home/grem3/mahquranapp/public/audio/abdul_basit_murattal/surah_001.mp3";
const WAVEFORM_DIR = "/home/grem3/mahquranapp/public/data/waveforms";
if (!fs.existsSync(WAVEFORM_DIR)) fs.mkdirSync(WAVEFORM_DIR, { recursive: true });
const OUT_PATH = path.join(WAVEFORM_DIR, "abdul_basit_murattal_surah_1.json");

const fileBuffer = fs.readFileSync(AUDIO_PATH);
const decoder = new MPEGDecoder();
await decoder.ready;
const { channelData, sampleRate } = decoder.decode(fileBuffer);
decoder.free();

const samples = channelData[0];
const duration = samples.length / sampleRate;

// 100 peaks per second
const windowSize = Math.floor(sampleRate / 100);
const peaks = [];
const rmsValues = [];

for (let i = 0; i < samples.length; i += windowSize) {
  let maxVal = 0;
  let sumSq = 0;
  let count = 0;
  for (let j = i; j < Math.min(i + windowSize, samples.length); j++) {
    const absVal = Math.abs(samples[j]);
    if (absVal > maxVal) maxVal = absVal;
    sumSq += absVal * absVal;
    count++;
  }
  const rms = count > 0 ? Math.sqrt(sumSq / count) : 0;
  peaks.push(Number(maxVal.toFixed(4)));
  rmsValues.push(Number(rms.toFixed(4)));
}

const waveformData = {
  reciter: "abdul_basit_murattal",
  surah: 1,
  duration: Number(duration.toFixed(3)),
  sampleRate,
  peaksPerSecond: 100,
  peaks,
  rms: rmsValues
};

fs.writeFileSync(OUT_PATH, JSON.stringify(waveformData, null, 2));
console.log(`✓ Successfully extracted ${peaks.length} waveform peaks across ${duration.toFixed(2)}s for Surah 1!`);
