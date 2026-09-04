// Convert Latin digits to authentic Eastern Arabic numerals (e.g. 1 -> ١)
export function toArabicNumerals(num: number): string {
  const digits = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
  return num
    .toString()
    .split("")
    .map(d => digits[parseInt(d, 10)] || d)
    .join("");
}

// Arabic diacritics / Tashkeel set for orthographic decomposition
export const ARABIC_DIACRITICS = new Set([
  "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651",
  "\u0652", "\u0653", "\u0654", "\u0655", "\u0670", "\u06DF", "\u06E0",
  "\u06E1", "\u06E2", "\u06E3", "\u06E4"
]);

// Split Arabic text into composite grapheme clusters (base letter + attached diacritics)
export function splitIntoGraphemes(text: string): string[] {
  const chunks: string[] = [];
  let cur = "";
  for (const char of text) {
    if (ARABIC_DIACRITICS.has(char)) {
      cur += char;
    } else {
      if (cur) chunks.push(cur);
      cur = char;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}
