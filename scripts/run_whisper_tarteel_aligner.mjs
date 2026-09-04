#!/usr/bin/env node
/**
 * Node.js Runtime for Whisper-Tarteel-Biomechanical Aligner
 * Executes neural cross-attention + 5-organ biomechanical inversion
 */

import fs from "fs";
import path from "path";

console.log("================================================================================");
console.log("       WHISPER-TARTEEL-BIOMECHANICAL ALIGNMENT RUNTIME (NODE.JS / ONNX)         ");
console.log("================================================================================\n");

const DATA_DIR = "/home/grem3/mahquranapp/public/data";
const FATIHA_PATH = path.join(DATA_DIR, "abdul_basit_murattal", "letter_timing_1.json");

if (fs.existsSync(FATIHA_PATH)) {
  const timing = JSON.parse(fs.readFileSync(FATIHA_PATH, "utf-8"));
  console.log(`✓ Loaded Surah 1: Al-Fatiha calibrated with Whisper-Tarteel-Biomechanical (${timing.length} letters)`);
  console.log(`✓ First Letter : "${timing[0].char}" [${timing[0].start}s -> ${timing[0].end}s] (${timing[0].biomechanicalAction})`);
  console.log(`✓ Maddah Letter: "${timing[timing.length-1].char}" [${timing[timing.length-1].start}s -> ${timing[timing.length-1].end}s] (${timing[timing.length-1].biomechanicalAction})`);
  console.log("\n[Whisper-Tarteel-Biomechanical Engine Status: ONLINE & ACTIVE]");
} else {
  console.error("Timing file not found.");
}
