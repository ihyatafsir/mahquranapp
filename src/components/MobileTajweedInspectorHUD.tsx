import React from "react";
import type { TimedWord, TimedLetter, LetterTiming } from "../types/quran";
import { getExactTajweedSign, type TajweedSignDetail } from "../utils/tajweedRules";

interface MobileTajweedInspectorHUDProps {
  activeWord: TimedWord | null;
  activeLetter: LetterTiming | TimedLetter | null;
  isPlaying?: boolean;
  hoveredTajweed?: TajweedSignDetail | null;
  hoveredWordText?: string | null;
}

export const MobileTajweedInspectorHUD: React.FC<MobileTajweedInspectorHUDProps> = ({
  activeWord,
  activeLetter,
  hoveredTajweed,
  hoveredWordText,
}) => {
  // If user is hovering over a word, prioritize displaying hovered info
  if (hoveredWordText && hoveredTajweed) {
    return (
      <div
        className="mobile-tajweed-hud-bar active hovered"
        style={{
          borderColor: hoveredTajweed.color,
          boxShadow: `0 0 16px ${hoveredTajweed.color}44`,
        }}
      >
        <div className="hud-left">
          <span
            className="hud-indicator-dot"
            style={{
              backgroundColor: hoveredTajweed.color,
              boxShadow: `0 0 10px ${hoveredTajweed.color}`,
            }}
          />
          <span className="hud-word-arabic" style={{ color: "#ffffff" }}>
            {hoveredWordText}
          </span>
          {hoveredTajweed.badge && (
            <span
              className="hud-rule-badge"
              style={{
                borderColor: hoveredTajweed.color,
                color: hoveredTajweed.color,
                backgroundColor: `${hoveredTajweed.color}22`,
              }}
            >
              {hoveredTajweed.badge}
            </span>
          )}
        </div>

        <div className="hud-right">
          <span className="hud-rule-name" style={{ color: "#e2e8f0" }}>
            {hoveredTajweed.ruleNameAr} ({hoveredTajweed.ruleNameEn})
          </span>
          {hoveredTajweed.harakatCount ? (
            <span
              className="hud-harakat-count"
              style={{ color: hoveredTajweed.color, borderColor: hoveredTajweed.color }}
            >
              ⏱ {hoveredTajweed.harakatCount} حركات
            </span>
          ) : null}
        </div>
      </div>
    );
  }

  if (!activeWord) {
    return (
      <div className="mobile-tajweed-hud-bar idle">
        <span className="hud-indicator-dot" />
        <span className="hud-label">TAJWEED MUSHAF ENGINE • Ready</span>
      </div>
    );
  }

  // Find active char or prominent tajweed char in word
  let tajweedChar = activeLetter ? activeLetter.char : "";
  let tajweed = getExactTajweedSign(tajweedChar);

  if (!tajweed.badge && activeWord.letters) {
    for (const l of activeWord.letters) {
      const t = getExactTajweedSign(l.char);
      if (t.badge && t.rule !== "normal" && t.rule !== "silent_wasl") {
        tajweed = t;
        tajweedChar = l.char;
        break;
      }
    }
  }

  return (
    <div
      className="mobile-tajweed-hud-bar active"
      style={{
        borderColor: tajweed.color,
        boxShadow: `0 0 16px ${tajweed.color}33`,
      }}
    >
      <div className="hud-left">
        <span
          className="hud-indicator-dot pulse"
          style={{
            backgroundColor: tajweed.color,
            boxShadow: `0 0 12px ${tajweed.color}`,
          }}
        />
        <span className="hud-word-arabic" style={{ color: "#ffffff" }}>
          {activeWord.arabic || activeWord.text}
        </span>
        {activeLetter?.char && (
          <span
            className="hud-letter-pill"
            style={{
              borderColor: `${tajweed.color}66`,
              color: tajweed.color,
              backgroundColor: `${tajweed.color}18`,
            }}
          >
            حرف: {activeLetter.char}
          </span>
        )}
        {tajweed.badge && (
          <span
            className="hud-rule-badge"
            style={{
              borderColor: tajweed.color,
              color: tajweed.color,
              backgroundColor: `${tajweed.color}22`,
            }}
          >
            {tajweed.badge}
          </span>
        )}
      </div>

      <div className="hud-right">
        <span className="hud-rule-name" style={{ color: "#cbd5e1" }}>
          {tajweed.ruleNameAr}
        </span>
        {tajweed.harakatCount ? (
          <span
            className="hud-harakat-count"
            style={{ color: tajweed.color, borderColor: tajweed.color }}
          >
            ⏱ {tajweed.harakatCount} حركات
          </span>
        ) : null}
      </div>
    </div>
  );
};
