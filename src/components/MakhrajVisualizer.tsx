import React from "react";

export interface FullVocalTractInfo {
  letter: string;
  name: string;
  transliteration: string;
  primaryOrgan: "Jawf" | "Halq" | "Lisan" | "Shafatan" | "Khayshum";
  primaryOrganAr: string;
  subLocationAr: string;
  subLocationEn: string;
  // Vocal Tract Coordinates (Sagittal SVG 0-100 x, 0-100 y)
  targetPoint: { x: number; y: number };
  lipState: "closed" | "rounded" | "neutral" | "teeth_on_lip" | "wide_open";
  tongueState: "flat" | "tip_alveolar" | "tip_dental" | "edge_lateral" | "back_velar" | "mid_palatal" | "pharyngeal";
  nasalActive: boolean;
  sifat: string[];
  isTafkheem: boolean;
  isQalqalah: boolean;
  isGhunnah: boolean;
}

export const COMPLETE_VOCAL_TRACT_DATA: Record<string, FullVocalTractInfo> = {
  // 1. AL-HALQ (The Throat)
  "ء": {
    letter: "ء", name: "Hamzah", transliteration: "ʾ",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "أقصى الحلق (عند الأوتار الصوتية)",
    subLocationEn: "Deepest throat / vocal cords",
    targetPoint: { x: 38, y: 78 },
    lipState: "neutral", tongueState: "flat", nasalActive: false,
    sifat: ["جهر", "شدة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ه": {
    letter: "ه", name: "Haa", transliteration: "h",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "أقصى الحلق (مع جريان النفس)",
    subLocationEn: "Deepest throat (open glottis)",
    targetPoint: { x: 38, y: 78 },
    lipState: "neutral", tongueState: "flat", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ع": {
    letter: "ع", name: "Ayn", transliteration: "ʿ",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "وسط الحلق (عند لسان المزمار)",
    subLocationEn: "Middle throat / Epiglottis",
    targetPoint: { x: 40, y: 68 },
    lipState: "neutral", tongueState: "pharyngeal", nasalActive: false,
    sifat: ["جهر", "بينية", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ح": {
    letter: "ح", name: "Haa", transliteration: "ḥ",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "وسط الحلق (مع حفيف النفس)",
    subLocationEn: "Middle throat (friction & breath)",
    targetPoint: { x: 40, y: 68 },
    lipState: "neutral", tongueState: "pharyngeal", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "غ": {
    letter: "غ", name: "Ghayn", transliteration: "gh",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "أدنى الحلق (عند اللهاة)",
    subLocationEn: "Upper throat near uvula",
    targetPoint: { x: 44, y: 58 },
    lipState: "rounded", tongueState: "back_velar", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استعلاء", "انفتاح"],
    isTafkheem: true, isQalqalah: false, isGhunnah: false
  },
  "خ": {
    letter: "خ", name: "Khaa", transliteration: "kh",
    primaryOrgan: "Halq", primaryOrganAr: "الحلق",
    subLocationAr: "أدنى الحلق (مع الحفيف والهمس)",
    subLocationEn: "Upper throat / friction",
    targetPoint: { x: 44, y: 58 },
    lipState: "rounded", tongueState: "back_velar", nasalActive: false,
    sifat: ["همس", "رخاوة", "استعلاء", "انفتاح"],
    isTafkheem: true, isQalqalah: false, isGhunnah: false
  },

  // 2. AL-LISAN (The Tongue)
  "ق": {
    letter: "ق", name: "Qaaf", transliteration: "q",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "أقصى اللسان مع الحنك اللحمي",
    subLocationEn: "Back of tongue against soft palate",
    targetPoint: { x: 48, y: 48 },
    lipState: "rounded", tongueState: "back_velar", nasalActive: false,
    sifat: ["جهر", "شدة", "استعلاء", "قلقلة"],
    isTafkheem: true, isQalqalah: true, isGhunnah: false
  },
  "ك": {
    letter: "ك", name: "Kaaf", transliteration: "k",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "أقصى اللسان مع الحنك اللحمي والعظمي",
    subLocationEn: "Back of tongue (below Qaaf)",
    targetPoint: { x: 52, y: 46 },
    lipState: "neutral", tongueState: "back_velar", nasalActive: false,
    sifat: ["همس", "شدة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ج": {
    letter: "ج", name: "Jeem", transliteration: "j",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "وسط اللسان مع الحنك الأعلى العظمي",
    subLocationEn: "Middle tongue against hard palate",
    targetPoint: { x: 58, y: 44 },
    lipState: "neutral", tongueState: "mid_palatal", nasalActive: false,
    sifat: ["جهر", "شدة", "استفال", "قلقلة"],
    isTafkheem: false, isQalqalah: true, isGhunnah: false
  },
  "ش": {
    letter: "ش", name: "Sheen", transliteration: "sh",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "وسط اللسان (مع انتشار الهواء / التفشي)",
    subLocationEn: "Middle tongue with Tafashshi",
    targetPoint: { x: 58, y: 44 },
    lipState: "neutral", tongueState: "mid_palatal", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "تفشي"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ي": {
    letter: "ي", name: "Yaa", transliteration: "y",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "وسط اللسان مع الحنك الأعلى",
    subLocationEn: "Middle tongue raised",
    targetPoint: { x: 58, y: 44 },
    lipState: "neutral", tongueState: "mid_palatal", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استفال", "لين"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ض": {
    letter: "ض", name: "Dhaad", transliteration: "ḍ",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "حافة اللسان مع الأضراس العليا (مع الاستطالة)",
    subLocationEn: "Edge of tongue against upper molars (Istitalah)",
    targetPoint: { x: 62, y: 46 },
    lipState: "rounded", tongueState: "edge_lateral", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استعلاء", "إطباق", "استطالة"],
    isTafkheem: true, isQalqalah: false, isGhunnah: false
  },
  "ل": {
    letter: "ل", name: "Laam", transliteration: "l",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "أدنى حافتي اللسان إلى منتهاها مع لثة الثنايا",
    subLocationEn: "Sides of tongue tip to upper gums",
    targetPoint: { x: 68, y: 40 },
    lipState: "neutral", tongueState: "tip_alveolar", nasalActive: false,
    sifat: ["جهر", "بينية", "استفال", "انحراف"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ن": {
    letter: "ن", name: "Noon", transliteration: "n",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان والخيشوم",
    subLocationAr: "طرف اللسان مع لثة الثنايا العليا + الغنة",
    subLocationEn: "Tongue tip on gums + Nasal cavity",
    targetPoint: { x: 70, y: 39 },
    lipState: "neutral", tongueState: "tip_alveolar", nasalActive: true,
    sifat: ["جهر", "بينية", "استفال", "غنة"],
    isTafkheem: false, isQalqalah: false, isGhunnah: true
  },
  "ر": {
    letter: "ر", name: "Raa", transliteration: "r",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان أدخل لظهره مع لثة الثنايا (تكرير لطيف)",
    subLocationEn: "Tongue tip with subtle vibration",
    targetPoint: { x: 69, y: 39 },
    lipState: "neutral", tongueState: "tip_alveolar", nasalActive: false,
    sifat: ["جهر", "بينية", "استفال", "تكرير"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ط": {
    letter: "ط", name: "Taa", transliteration: "ṭ",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أصول الثنايا العليا (مع الإطباق)",
    subLocationEn: "Tongue tip at roots of incisors (Itbaq)",
    targetPoint: { x: 74, y: 41 },
    lipState: "rounded", tongueState: "tip_dental", nasalActive: false,
    sifat: ["جهر", "شدة", "استعلاء", "إطباق", "قلقلة"],
    isTafkheem: true, isQalqalah: true, isGhunnah: false
  },
  "د": {
    letter: "د", name: "Daal", transliteration: "d",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أصول الثنايا العليا",
    subLocationEn: "Tongue tip at roots of incisors",
    targetPoint: { x: 74, y: 41 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["جهر", "شدة", "استفال", "قلقلة"],
    isTafkheem: false, isQalqalah: true, isGhunnah: false
  },
  "ت": {
    letter: "ت", name: "Taa", transliteration: "t",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أصول الثنايا العليا (همس)",
    subLocationEn: "Tongue tip at incisors with breath",
    targetPoint: { x: 74, y: 41 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["همس", "شدة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ص": {
    letter: "ص", name: "Saad", transliteration: "ṣ",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان فوق الثنايا السفلى (صفير وإطباق)",
    subLocationEn: "Tongue tip over lower teeth (Safeer)",
    targetPoint: { x: 75, y: 45 },
    lipState: "rounded", tongueState: "tip_dental", nasalActive: false,
    sifat: ["همس", "رخاوة", "استعلاء", "إطباق", "صفير"],
    isTafkheem: true, isQalqalah: false, isGhunnah: false
  },
  "س": {
    letter: "س", name: "Seen", transliteration: "s",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان فوق الثنايا السفلى (صفير وهمس)",
    subLocationEn: "Tongue tip over lower teeth (Safeer)",
    targetPoint: { x: 75, y: 45 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "صفير"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ز": {
    letter: "ز", name: "Zaay", transliteration: "z",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان فوق الثنايا السفلى (صفير وجهر)",
    subLocationEn: "Tongue tip over lower teeth (Jahr)",
    targetPoint: { x: 75, y: 45 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استفال", "صفير"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ظ": {
    letter: "ظ", name: "Dhaa", transliteration: "ẓ",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أطراف الثنايا العليا (إطباق)",
    subLocationEn: "Tongue tip between incisors (Itbaq)",
    targetPoint: { x: 78, y: 43 },
    lipState: "rounded", tongueState: "tip_dental", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استعلاء", "إطباق"],
    isTafkheem: true, isQalqalah: false, isGhunnah: false
  },
  "ذ": {
    letter: "ذ", name: "Dhaal", transliteration: "dh",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أطراف الثنايا العليا",
    subLocationEn: "Tongue tip between incisors",
    targetPoint: { x: 78, y: 43 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ث": {
    letter: "ث", name: "Thaa", transliteration: "th",
    primaryOrgan: "Lisan", primaryOrganAr: "اللسان",
    subLocationAr: "طرف اللسان مع أطراف الثنايا العليا (همس)",
    subLocationEn: "Tongue tip between incisors with breath",
    targetPoint: { x: 78, y: 43 },
    lipState: "neutral", tongueState: "tip_dental", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },

  // 3. ASH-SHAFATAN (The Lips)
  "ف": {
    letter: "ف", name: "Faa", transliteration: "f",
    primaryOrgan: "Shafatan", primaryOrganAr: "الشفتان",
    subLocationAr: "بطن الشفة السفلى مع أطراف الثنايا العليا",
    subLocationEn: "Inside lower lip to upper incisors",
    targetPoint: { x: 84, y: 44 },
    lipState: "teeth_on_lip", tongueState: "flat", nasalActive: false,
    sifat: ["همس", "رخاوة", "استفال", "انفتاح"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },
  "ب": {
    letter: "ب", name: "Baa", transliteration: "b",
    primaryOrgan: "Shafatan", primaryOrganAr: "الشفتان",
    subLocationAr: "انطباق الشفتين بقوة مع الجهر والشدة",
    subLocationEn: "Firm closure of both lips (Explosion)",
    targetPoint: { x: 88, y: 46 },
    lipState: "closed", tongueState: "flat", nasalActive: false,
    sifat: ["جهر", "شدة", "استفال", "قلقلة"],
    isTafkheem: false, isQalqalah: true, isGhunnah: false
  },
  "م": {
    letter: "م", name: "Meem", transliteration: "m",
    primaryOrgan: "Shafatan", primaryOrganAr: "الشفتان والخيشوم",
    subLocationAr: "انطباق الشفتين بلطف مع الغنة من الخيشوم",
    subLocationEn: "Gentle lip closure + Nasal resonance",
    targetPoint: { x: 88, y: 46 },
    lipState: "closed", tongueState: "flat", nasalActive: true,
    sifat: ["جهر", "بينية", "استفال", "غنة"],
    isTafkheem: false, isQalqalah: false, isGhunnah: true
  },
  "و": {
    letter: "و", name: "Waaw", transliteration: "w",
    primaryOrgan: "Shafatan", primaryOrganAr: "الشفتان",
    subLocationAr: "انضمام الشفتين مع فرجة يسيرة",
    subLocationEn: "Rounding and protruding of lips",
    targetPoint: { x: 90, y: 46 },
    lipState: "rounded", tongueState: "flat", nasalActive: false,
    sifat: ["جهر", "رخاوة", "استفال", "لين"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  },

  // 4. AL-JAWF (Oral Cavity / Madd)
  "ا": {
    letter: "ا", name: "Alif Madd", transliteration: "ā",
    primaryOrgan: "Jawf", primaryOrganAr: "الجوف",
    subLocationAr: "الخلاء الممتد من الحلق عبر تجويف الفم",
    subLocationEn: "Open oral cavity & throat radiation",
    targetPoint: { x: 60, y: 55 },
    lipState: "wide_open", tongueState: "flat", nasalActive: false,
    sifat: ["جهر", "رخاوة", "خفاء", "مد"],
    isTafkheem: false, isQalqalah: false, isGhunnah: false
  }
};

interface MakhrajVisualizerProps {
  currentChar: string | null;
  isPlaying: boolean;
}

export const MakhrajVisualizer: React.FC<MakhrajVisualizerProps> = ({ currentChar, isPlaying }) => {
  if (!currentChar) return null;

  const cleanChar = currentChar.replace(/[\u064B-\u065F\u0670\u06E1\u06DF-\u06E3]/g, "");
  const baseLetter = cleanChar.length > 0 ? cleanChar[0] : currentChar[0];
  const info = COMPLETE_VOCAL_TRACT_DATA[baseLetter] || COMPLETE_VOCAL_TRACT_DATA["ا"];

  return (
    <div className={`makhraj-panel ${isPlaying ? "is-active" : ""}`}>
      {/* Header */}
      <div className="makhraj-header">
        <div className="makhraj-header-title">
          <span className="makhraj-badge">الأعضاء الصوتية الخمسة والمشافهة (5 Vocal Organs & Lips)</span>
          <span className="makhraj-organ-chip">{info.primaryOrganAr} • {info.primaryOrgan}</span>
        </div>
        <span className="makhraj-char-big">{currentChar}</span>
      </div>

      <div className="makhraj-body-grid">
        {/* Sagittal Cross-Section Anatomical Diagram */}
        <div className="vocal-tract-diagram-card">
          <svg viewBox="0 0 100 90" className="sagittal-vocal-tract-svg">
            <defs>
              {/* Radial glow for active articulation target */}
              <radialGradient id="targetGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#00ffaa" stopOpacity="1" />
                <stop offset="60%" stopColor="#00ffaa" stopOpacity="0.5" />
                <stop offset="100%" stopColor="#00ffaa" stopOpacity="0" />
              </radialGradient>
            </defs>

            {/* Head Silhouette Outline (Throat -> Jaw -> Mouth -> Nose) */}
            <path
              d="M 32,88 Q 30,70 34,55 Q 38,40 50,30 Q 65,20 80,25 Q 92,30 92,42 L 86,45 Q 88,48 84,52 Q 78,56 70,60 Q 60,65 52,70 Q 42,75 40,88 Z"
              fill="rgba(15, 23, 42, 0.6)"
              stroke="rgba(56, 189, 248, 0.3)"
              strokeWidth="1.2"
            />

            {/* Nasal Cavity (Al-Khayshum) */}
            <path
              d="M 60,26 Q 75,22 84,32 Q 75,36 62,34 Z"
              fill={info.nasalActive ? "rgba(34, 197, 94, 0.3)" : "rgba(255, 255, 255, 0.04)"}
              stroke={info.nasalActive ? "#22c55e" : "rgba(148, 163, 184, 0.2)"}
              strokeWidth="1"
            />
            {info.nasalActive && (
              <text x="73" y="30" fill="#4ade80" fontSize="3" textAnchor="middle">غنة (Nasal)</text>
            )}

            {/* Hard & Soft Palate (Al-Hanak) */}
            <path d="M 48,46 Q 60,38 76,40" fill="none" stroke="#94a3b8" strokeWidth="1.5" />

            {/* Tongue Body (Al-Lisan) with dynamic curvature */}
            <path
              d={
                info.tongueState === "tip_alveolar"
                  ? "M 40,70 Q 50,55 60,50 Q 66,42 70,40 Q 65,60 48,72 Z"
                  : info.tongueState === "tip_dental"
                  ? "M 40,70 Q 50,55 62,48 Q 72,42 76,42 Q 68,60 48,72 Z"
                  : info.tongueState === "back_velar"
                  ? "M 40,70 Q 46,48 50,47 Q 60,54 70,52 Q 62,64 48,72 Z"
                  : info.tongueState === "mid_palatal"
                  ? "M 40,70 Q 50,44 58,43 Q 66,48 72,50 Q 62,64 48,72 Z"
                  : "M 40,70 Q 50,60 62,56 Q 72,54 76,52 Q 65,65 48,72 Z" // flat / neutral
              }
              fill="rgba(255, 94, 126, 0.4)"
              stroke="#ff5e7e"
              strokeWidth="1.2"
            />

            {/* Upper & Lower Incisors (Teeth) */}
            <rect x="79" y="39" width="3" height="4" rx="0.5" fill="#ffffff" stroke="#cbd5e1" strokeWidth="0.5" />
            <rect x="79" y="47" width="3" height="4" rx="0.5" fill="#ffffff" stroke="#cbd5e1" strokeWidth="0.5" />

            {/* Lips (Ash-Shafatan) */}
            {info.lipState === "closed" ? (
              <path d="M 85,42 Q 88,45 85,48 Z" fill="#ff0055" stroke="#ff0055" strokeWidth="1" />
            ) : info.lipState === "rounded" ? (
              <ellipse cx="86" cy="45" rx="3" ry="4" fill="none" stroke="#00ffaa" strokeWidth="1.5" />
            ) : (
              <path d="M 84,41 Q 87,43 85,45 M 84,49 Q 87,47 85,45" fill="none" stroke="#ff0055" strokeWidth="1" />
            )}

            {/* Active Articulation Point Laser Glow */}
            <circle cx={info.targetPoint.x} cy={info.targetPoint.y} r="5" fill="url(#targetGlow)" />
            <circle cx={info.targetPoint.x} cy={info.targetPoint.y} r="2" fill="#ffffff" />

            {/* Vocal Tract Organ Labels */}
            <text x="32" y="80" fill="#38bdf8" fontSize="2.8">الحلق</text>
            <text x="56" y="62" fill="#ff5e7e" fontSize="2.8">اللسان</text>
            <text x="89" y="52" fill="#facc15" fontSize="2.8">الشفتان</text>
          </svg>
          <span className="diagram-sublabel">مقطع تشريحي للنطق (Sagittal Vocal Tract Articulation)</span>
        </div>

        {/* Anatomical Description & Tajweed Sifat Panel */}
        <div className="makhraj-details-panel">
          <div className="makhraj-subloc-ar">{info.subLocationAr}</div>
          <div className="makhraj-subloc-en">{info.subLocationEn}</div>

          <div className="organ-status-pills">
            <span className="organ-pill">👄 الشفتان: {info.lipState.replace("_", " ")}</span>
            <span className="organ-pill">👅 اللسان: {info.tongueState.replace("_", " ")}</span>
            <span className={`organ-pill ${info.nasalActive ? "nasal-glow" : ""}`}>
              👃 الخيشوم: {info.nasalActive ? "غنة نشطة" : "مغلق"}
            </span>
          </div>

          <div className="sifat-chips-row">
            {info.sifat.map((s, idx) => (
              <span key={idx} className="sifat-chip">{s}</span>
            ))}
            {info.isTafkheem && <span className="sifat-chip tafkheem">تفخيم (Tafkheem)</span>}
            {info.isQalqalah && <span className="sifat-chip qalqalah">قلقلة (Qalqalah)</span>}
            {info.isGhunnah && <span className="sifat-chip ghunnah">غنة (Ghunnah)</span>}
          </div>
        </div>
      </div>
    </div>
  );
};
