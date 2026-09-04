export type TajweedRuleType = 
  | "madd_lazim_6"
  | "madd_wajib_muttasil_4_5"
  | "madd_jaiz_munfasil_4_5"
  | "madd_tabiee_2"
  | "ghunnah_mushaddadah_2"
  | "ikhfa_idgham_ghunnah"
  | "iqlab"
  | "qalqalah"
  | "tafkheem"
  | "silent_wasl"
  | "normal";

export interface TajweedSignDetail {
  rule: TajweedRuleType;
  ruleNameAr: string;
  ruleNameEn: string;
  color: string; // Standard Mushaf Tajweed color
  badge: string;
  harakatCount?: number;
  symbol: string;
}

export function getExactTajweedSign(chunk: string, nextChunk?: string, _prevChunk?: string): TajweedSignDetail {
  const base = chunk[0];
  const hasMaddah = chunk.includes("ٓ") || chunk.includes("\u0653") || chunk.includes("\u06E4");
  const hasShaddah = chunk.includes("ّ") || chunk.includes("\u0651");
  const hasSukoon = chunk.includes("ْ") || chunk.includes("\u0652") || chunk.includes("\u06E1");
  const hasSilentCircle = chunk.includes("۟") || chunk.includes("\u06DF");
  const hasDaggerAlif = chunk.includes("ٰ") || chunk.includes("\u0670");
  const hasIqlabMeem = chunk.includes("ۭ") || chunk.includes("ۢ") || chunk.includes("\u06E2") || chunk.includes("\u06ED");

  // 1. Silent Letters (همزة الوصل ٱ, اللام الشمسية, ألف التفريق ۟) -> Standard Mushaf Grey #94a3b8
  if (hasSilentCircle || base === "\u0671" || (base === "ل" && nextChunk && (nextChunk.includes("ّ") || nextChunk.includes("\u0651")) && !hasSukoon)) {
    return {
      rule: "silent_wasl",
      ruleNameAr: "همزة وصل / حرف لا يُنطق",
      ruleNameEn: "Silent / Elided Letter",
      color: "#94a3b8",
      badge: "صلة ٱ",
      symbol: "ٱ"
    };
  }

  // 2. Madd Lazim (6 Harakat: ضَّآلِّينَ) -> Standard Mushaf Crimson / Deep Red #ef4444
  if (hasMaddah && (hasShaddah || (nextChunk && (nextChunk.includes("ّ") || nextChunk.includes("\u0651"))))) {
    return {
      rule: "madd_lazim_6",
      ruleNameAr: "مد لازم (6 حركات)",
      ruleNameEn: "Madd Lazim (6 Harakat)",
      color: "#f43f5e", // Radiant Crimson
      badge: "مد لازم 6ح ~",
      harakatCount: 6,
      symbol: "ٓ"
    };
  }

  // 3. Madd Wajib / Jaiz (4-5 Harakat: جَآءَ, السَّمَآءِ) -> Standard Mushaf Red-Orange #f97316
  if (hasMaddah) {
    return {
      rule: "madd_wajib_muttasil_4_5",
      ruleNameAr: "مد واجب/جائز (4-5 حركات)",
      ruleNameEn: "Madd Wajib/Jaiz (4-5 Harakat)",
      color: "#fb923c",
      badge: "مد 4-5ح ~",
      harakatCount: 5,
      symbol: "ٓ"
    };
  }

  // 4. Ghunnah on Noon/Meem Mushaddadah (نّ, مّ) -> Standard Mushaf Emerald Green #10b981
  if (hasShaddah && (base === "ن" || base === "م")) {
    return {
      rule: "ghunnah_mushaddadah_2",
      ruleNameAr: "غنة الحرف المشدد (2 حركة)",
      ruleNameEn: "Ghunnah (2 Harakat)",
      color: "#10b981", // Luminous Emerald Green
      badge: "غنة 2ح ّ",
      harakatCount: 2,
      symbol: "ّ"
    };
  }

  // 5. Iqlab (ۢ) -> Standard Mushaf Sea Green #34d399
  if (hasIqlabMeem) {
    return {
      rule: "iqlab",
      ruleNameAr: "إقلاب النون/التنوين ميماً",
      ruleNameEn: "Iqlab (Nasal Meem)",
      color: "#34d399",
      badge: "إقلاب ۢ",
      symbol: "ۢ"
    };
  }

  // 6. Qalqalah (ق ط ب ج د with Sukoon) -> Standard Mushaf Cyan/Sky Blue #06b6d4
  if (hasSukoon && "قطبجد".includes(base)) {
    return {
      rule: "qalqalah",
      ruleNameAr: "قلقلة",
      ruleNameEn: "Qalqalah (Echoing Burst)",
      color: "#06b6d4",
      badge: "قلقلة ⚡",
      symbol: "ْ"
    };
  }

  // 7. Tafkheem & Istitalah (خص ضغط قظ) -> Standard Mushaf Royal Blue/Purple #818cf8
  if ("خصضغطقظ".includes(base)) {
    const isDhad = base === "ض";
    return {
      rule: "tafkheem",
      ruleNameAr: isDhad ? "استطالة وتفخيم الضاد" : "حرف مفخم (استعلاء)",
      ruleNameEn: isDhad ? "Dhad Istitalah" : "Tafkheem",
      color: "#a78bfa",
      badge: isDhad ? "استطالة ض" : "تفخيم",
      symbol: "▲"
    };
  }

  // 8. Madd Tabiee / Dagger Alif (2 Harakat) -> Standard Mushaf Gold/Amber #f59e0b
  if (hasDaggerAlif || (("اوية".includes(base) || base === "ى") && !hasSukoon && chunk.length === 1)) {
    return {
      rule: "madd_tabiee_2",
      ruleNameAr: "مد طبيعي (2 حركة)",
      ruleNameEn: "Natural Madd (2 Harakat)",
      color: "#f59e0b",
      badge: "مد طبيعي 2ح",
      harakatCount: 2,
      symbol: "ٰ"
    };
  }

  return {
    rule: "normal",
    ruleNameAr: "حرف مرقق",
    ruleNameEn: "Standard Tarqeeq",
    color: "#f8fafc", // Crisp White
    badge: "",
    symbol: ""
  };
}
