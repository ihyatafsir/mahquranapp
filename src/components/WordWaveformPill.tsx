import React from "react";
import type { TimedWord, TimedLetter, LetterTiming } from "../types/quran";
import { getExactTajweedSign, type TajweedSignDetail } from "../utils/tajweedRules";

interface WordWaveformPillProps {
  word: TimedWord;
  isWordActive: boolean;
  isWordPast: boolean;
  currentTime: number;
  activeLetter?: LetterTiming | TimedLetter | null;
  isAuditioning?: boolean;
  enableTajweedColors?: boolean;
  onWordClick: (word: TimedWord) => void;
  onWordHover?: (wordText: string | null, tajweed?: TajweedSignDetail) => void;
}

export const WordWaveformPill: React.FC<WordWaveformPillProps> = ({
  word,
  isWordActive,
  isWordPast,
  currentTime,
  activeLetter,
  isAuditioning = false,
  enableTajweedColors = true,
  onWordClick,
  onWordHover,
}) => {
  // -40ms Psychoacoustic Lead-Bias to align auditory & visual processing
  const effectiveTime = currentTime + 0.040;

  const activeLetterIdx =
    isWordActive && activeLetter && (activeLetter as any).wordIdx === word.globalWordIdx
      ? ((activeLetter as any).charIdxInWord ?? 0)
      : -1;

  const activeLetterFrac =
    isWordActive && activeLetter && activeLetter.end > activeLetter.start
      ? Math.max(
          0,
          Math.min(
            1,
            (effectiveTime - activeLetter.start) /
              Math.max(0.01, activeLetter.end - activeLetter.start)
          )
        )
      : 0;

  // Prominent Tajweed rule in this word
  const prominentTajweed = React.useMemo(() => {
    for (let i = 0; i < word.letters.length; i++) {
      const l = word.letters[i];
      const nxt = word.letters[i + 1]?.char;
      const prv = word.letters[i - 1]?.char;
      const t = getExactTajweedSign(l.char, nxt, prv);
      if (t.rule !== "normal" && t.rule !== "silent_wasl") {
        return t;
      }
    }
    return getExactTajweedSign(word.letters[0]?.char || "");
  }, [word.letters]);

  return (
    <span
      className={`mushaf-word-flow ${isWordActive ? "word-active" : ""} ${
        isWordPast ? "word-past" : ""
      } ${isAuditioning ? "word-auditioning" : ""}`}
      onClick={e => {
        e.stopPropagation();
        onWordClick(word);
      }}
      onMouseEnter={() => {
        if (onWordHover) {
          onWordHover(word.arabic || word.text, prominentTajweed);
        }
      }}
      onMouseLeave={() => {
        if (onWordHover) {
          onWordHover(null);
        }
      }}
      title={`Word #${word.globalWordIdx + 1} (${word.start.toFixed(2)}s - ${word.end.toFixed(2)}s) • Click to play word only`}
    >
      {word.letters.map((letter, idx) => {
        const isLetterActive = isWordActive && activeLetterIdx === idx;
        const isLetterPast =
          isWordPast || (isWordActive && activeLetterIdx > idx);

        const letterProgress = isLetterPast
          ? 100
          : isLetterActive
          ? Math.round(activeLetterFrac * 100)
          : 0;

        const nxt = word.letters[idx + 1]?.char;
        const prv = word.letters[idx - 1]?.char;
        const tajweed = getExactTajweedSign(letter.char, nxt, prv);
        const resolvedColor = enableTajweedColors ? tajweed.color : "#00ff88";

        return (
          <span
            key={idx}
            className={`mushaf-char-flow ${isLetterActive ? "char-active" : ""} ${
              isLetterPast ? "char-past" : ""
            }`}
            style={{
              "--letter-fill": `${letterProgress}%`,
              "--tajweed-color": resolvedColor,
            } as React.CSSProperties}
          >
            {letter.char}
          </span>
        );
      })}
      {isAuditioning && <span className="audition-badge">🔊</span>}
    </span>
  );
};

