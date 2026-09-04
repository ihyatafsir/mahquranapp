import React, { useMemo } from "react";
import type { LetterTiming, TimedLetter } from "../types/quran";
import { getVoicefingerprint } from "../engine/ReciterVoicefingerprint";

interface BiomechanicalVocalHUDProps {
  currentLetter: LetterTiming | TimedLetter | null;
  isPlaying: boolean;
  reciterId?: string;
}

export const BiomechanicalVocalHUD: React.FC<BiomechanicalVocalHUDProps> = ({
  currentLetter,
  isPlaying,
  reciterId = "abdul_basit_murattal",
}) => {
  const voicefinger = useMemo(() => getVoicefingerprint(reciterId), [reciterId]);
  const char = currentLetter ? currentLetter.char : "";
  const baseChar = char ? char.replace(/[\u064B-\u065F\u0670\u06D6-\u06ED]/g, "") : "";

  // Compute physiological biomechanical state
  const bioState = useMemo(() => {
    if (!isPlaying || !char) {
      return {
        lungs: "Neutral Breathing",
        throat: "Open Airway",
        tongueZone: "Resting",
        tongueAction: "Centralized in Oral Cavity",
        lips: "Parted",
        lipsAction: "Neutral Position",
        nasal: "Closed Velum",
        organName: "Silence / Breath Pause",
        organAr: "الاستراحة والتنفس",
        color: "#64748b"
      };
    }

    const hasMaddah = char.includes("ٓ") || char.includes("\u0653");
    // hasShaddah tracked via baseChar

    // Madd Lazim / Jawf
    if (hasMaddah || baseChar === "ا" || baseChar === "ى") {
      return {
        lungs: "Sustained Exhalation (Ps = 1.4 kPa)",
        throat: "Harmonic Glottal Phonation",
        tongueZone: "Low Flat (Al-Jawf)",
        tongueAction: "Wide Acoustic Resonance Corridor",
        lips: "Open Rounded",
        lipsAction: "Maximum Forward Radiation",
        nasal: "Velum Sealed",
        organName: "Oral Cavity (Al-Jawf) • Madd Wave",
        organAr: "الجوف • مد الصوت",
        color: "#00f0ff"
      };
    }

    // Bilabial Lips (Baa, Meem)
    if (baseChar === "م" || baseChar === "ب") {
      return {
        lungs: "Controlled Subglottal Build-up",
        throat: "Voiced Glottal Vibration",
        tongueZone: "Flat Resting",
        tongueAction: "Passive",
        lips: "Complete Bilabial Occlusion (Al-Itbaq)",
        lipsAction: "Upper & Lower Lips Compressed Tight",
        nasal: baseChar === "م" ? "Velum Lowered (Active Ghunnah)" : "Velum Sealed",
        organName: "Lips (Ash-Shafatan) • Bilabial Seal",
        organAr: "الشفتان • إطباق الشفتين",
        color: "#00ffaa"
      };
    }

    // Labiodental (Faa)
    if (baseChar === "ف") {
      return {
        lungs: "Continuous Fricative Exhalation",
        throat: "Unvoiced Whisper Flow (Hams)",
        tongueZone: "Resting Low",
        tongueAction: "Passive",
        lips: "Labiodental Incisor Contact",
        lipsAction: "Upper Central Incisors touching Wet Lower Lip",
        nasal: "Velum Sealed",
        organName: "Lips & Teeth • Labiodental",
        organAr: "الشفتان والأسنان • مخرج الفاء",
        color: "#38bdf8"
      };
    }

    // Dhaad (حافة اللسان - Lateral Molar Edge Istitalah)
    if (baseChar === "ض") {
      return {
        lungs: "High Lateral Subglottal Pressure",
        throat: "Voiced Heavy Pharyngeal (Tafkheem)",
        tongueZone: "Lateral Edges (Haffat al-Lisan)",
        tongueAction: "Tongue Edge pressed against Upper Molars (Istitalah)",
        lips: "Slight Protrusion",
        lipsAction: "Passive Support",
        nasal: "Velum Sealed",
        organName: "Tongue Lateral Edge (Al-Istitalah)",
        organAr: "حافة اللسان • الاستطالة والرخاوة",
        color: "#f59e0b"
      };
    }

    // Throat Letters (Halq)
    if ("ءهعحغخ".includes(baseChar)) {
      return {
        lungs: "Deep Respiratory Core Airflow",
        throat: "Active Pharyngeal & Laryngeal Constriction",
        tongueZone: "Retracted Root (Aqsa al-Lisan)",
        tongueAction: "Epiglottic Tension towards Pharynx",
        lips: "Neutral Parted",
        lipsAction: "Open Flow",
        nasal: "Velum Sealed",
        organName: "Throat & Larynx (Al-Halq)",
        organAr: "الحلق والحنجرة • مخرج الحروف الحلقية",
        color: "#a855f7"
      };
    }

    // Tongue Tip Alveolar / Dental (Noon, Raa, Laam, Daal, Taa, Seen, Saad)
    if ("نرلادتطصسزظذثجشيقك".includes(baseChar)) {
      const isNasal = baseChar === "ن";
      return {
        lungs: "Active Articulatory Exhalation",
        throat: "Voiced Glottal Train",
        tongueZone: "Tongue Tip & Blade (Taraf al-Lisan)",
        tongueAction: "Dynamic Alveolar & Dental Ridge Tap",
        lips: "Slight Jaw Drop",
        lipsAction: "Focused Acoustic Projection",
        nasal: isNasal ? "Velum Lowered (Active Ghunnah)" : "Velum Sealed",
        organName: "Tongue Tip & Blade (Al-Lisan)",
        organAr: "طرف اللسان واللثة العليا",
        color: "#10b981"
      };
    }

    return {
      lungs: "Active Breath Support",
      throat: "Voiced Phonation",
      tongueZone: "Active Vocalization",
      tongueAction: "Dynamic Vowel Shaping",
      lips: "Open Acoustic Radiator",
      lipsAction: "Slightly Rounded",
      nasal: "Velum Sealed",
      organName: "Vocal Tract",
      organAr: "المجرى الصوتي",
      color: "#00ffaa"
    };
  }, [isPlaying, char, baseChar]);

  return (
    <div className="biomechanical-hud-container" dir="ltr">
      <div className="hud-header">
        <div className="hud-title-badge">
          <span className="hud-pulse-dot" style={{ backgroundColor: bioState.color }} />
          <span>BIOMECHANICAL VOCAL APPARATUS • المشافهة والتلقي</span>
        </div>
        <div className="hud-organ-badge" style={{ borderColor: bioState.color, color: bioState.color }}>
          {bioState.organAr} ({bioState.organName})
        </div>
      </div>

      <div className="hud-body-grid">
        {/* 🫁 1. Lungs & Respiration */}
        <div className="hud-metric-card">
          <div className="hud-card-label">🫁 Lungs & Diaphragm (الصدر)</div>
          <div className="hud-card-value">{bioState.lungs}</div>
          <div className="hud-progress-bar">
            <div
              className="hud-progress-fill"
              style={{
                width: isPlaying ? "85%" : "20%",
                background: "linear-gradient(90deg, #3b82f6, #00f0ff)",
              }}
            />
          </div>
        </div>

        {/* 🗣️ 2. Throat & Larynx */}
        <div className="hud-metric-card">
          <div className="hud-card-label">🗣️ Throat & Glottis (الحلق)</div>
          <div className="hud-card-value">{bioState.throat}</div>
          <div className="hud-progress-bar">
            <div
              className="hud-progress-fill"
              style={{
                width: isPlaying ? "90%" : "15%",
                background: "linear-gradient(90deg, #8b5cf6, #a855f7)",
              }}
            />
          </div>
        </div>

        {/* 👅 3. Tongue Zone */}
        <div className="hud-metric-card highlight-card" style={{ borderColor: bioState.color }}>
          <div className="hud-card-label">👅 Tongue Muscle (اللسان)</div>
          <div className="hud-card-value" style={{ color: bioState.color, fontWeight: 700 }}>
            {bioState.tongueZone}
          </div>
          <div className="hud-sub-desc">{bioState.tongueAction}</div>
        </div>

        {/* 👄 4. Lips & Jaw */}
        <div className="hud-metric-card">
          <div className="hud-card-label">👄 Lips & Jaw (الشفتان)</div>
          <div className="hud-card-value">{bioState.lips}</div>
          <div className="hud-sub-desc">{bioState.lipsAction}</div>
        </div>

        {/* 👃 5. Nasal Resonance */}
        <div className="hud-metric-card">
          <div className="hud-card-label">👃 Nasal Cavity (الخيشوم)</div>
          <div className="hud-card-value" style={{ color: bioState.nasal.includes("Ghunnah") ? "#00ffaa" : "#94a3b8" }}>
            {bioState.nasal}
          </div>
          <div className="hud-progress-bar">
            <div
              className="hud-progress-fill"
              style={{
                width: bioState.nasal.includes("Ghunnah") ? "100%" : "5%",
                background: "#00ffaa",
                boxShadow: bioState.nasal.includes("Ghunnah") ? "0 0 8px #00ffaa" : "none",
              }}
            />
          </div>
        </div>
      </div>

      {/* Reciter Tajweed Voicefingerprint Telemetry */}
      <div className="voicefinger-telemetry-bar">
        <div className="vf-header">
          <span className="vf-icon">🎙️</span>
          <span className="vf-title">Reciter Tajweed Voicefingerprint:</span>
          <span className="vf-name">{voicefinger.name}</span>
          <span className="vf-style-badge">{voicefinger.style}</span>
        </div>
        <div className="vf-stats-row">
          <div className="vf-stat">
            <span className="vf-stat-lbl">Mora Tempo:</span>
            <span className="vf-stat-val">{voicefinger.baseMoraMs}ms / حركة</span>
          </div>
          <div className="vf-stat">
            <span className="vf-stat-lbl">Madd Lazim:</span>
            <span className="vf-stat-val">{voicefinger.maddLazimWeight}x</span>
          </div>
          <div className="vf-stat">
            <span className="vf-stat-lbl">Ghunnah Dome:</span>
            <span className="vf-stat-val">{voicefinger.ghunnahWeight}x</span>
          </div>
          <div className="vf-stat">
            <span className="vf-stat-lbl">Qalqalah Rebound:</span>
            <span className="vf-stat-val">{voicefinger.qalqalahBounceMs * 1000}ms</span>
          </div>
          <div className="vf-stat">
            <span className="vf-stat-lbl">Silent Suppression:</span>
            <span className="vf-stat-val">&le; 5ms</span>
          </div>
        </div>
      </div>
    </div>
  );
};
