import React, { useRef, useEffect, useCallback, useMemo, useState } from "react";
import type { Verse, LetterTiming } from "../types/quran";
import { getExactTajweedSign, type TajweedSignDetail } from "../utils/tajweedRules";
import { splitIntoGraphemes, toArabicNumerals } from "../utils/arabic";

interface FluidMushafCanvasProps {
  verses: Verse[];
  letterTiming: LetterTiming[];
  currentTime: number;
  isPlaying: boolean;
  activeVerseIdx: number;
  enableTajweedColors?: boolean;
  autoScroll?: boolean;
  onSeek: (time: number) => void;
  onWordHover?: (wordText: string | null, tajweed?: TajweedSignDetail) => void;
}

interface LetterBoundary {
  char: string;
  start: number;
  end: number;
  prefixPixelWidth: number; // Pixel width of word prefix ending at this letter
  tajweed: TajweedSignDetail;
}

interface WordWithLetterBoundaries {
  wordIdx: number;
  wordText: string;
  totalWidth: number;
  start: number;
  end: number;
  letters: LetterBoundary[];
  tajweed: TajweedSignDetail;
  // Computed layout bounds on canvas
  renderedX?: number;
  renderedY?: number;
  renderedWidth?: number;
  renderedHeight?: number;
}

interface WrappedLine {
  vIdx: number;
  ayahNum: number;
  isLastLineOfVerse: boolean;
  words: WordWithLetterBoundaries[];
  lineArabicText: string;
  totalWidth: number;
  start: number;
  end: number;
  lineY?: number;
  lineStartX?: number;
}

export const FluidMushafCanvas: React.FC<FluidMushafCanvasProps> = ({
  verses,
  letterTiming,
  currentTime,
  isPlaying,
  activeVerseIdx,
  enableTajweedColors = true,
  autoScroll = true,
  onSeek,
  onWordHover,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cachedLinesRef = useRef<WrappedLine[]>([]);
  const [hoveredWord, setHoveredWord] = useState<WordWithLetterBoundaries | null>(null);

  // Map letters by verse index
  const verseLettersMap = useMemo(() => {
    const map = new Map<number, LetterTiming[]>();
    if (!letterTiming || letterTiming.length === 0) return map;

    for (const lt of letterTiming) {
      const vIdx = lt.verseIdx !== undefined ? lt.verseIdx : (lt.ayah ? lt.ayah - 1 : 0);
      if (!map.has(vIdx)) {
        map.set(vIdx, []);
      }
      map.get(vIdx)!.push(lt);
    }
    return map;
  }, [letterTiming]);

  // Compute Layout with Connected Words, Prefix Sub-Pixel Boundaries, and Multi-Line Wrapping
  const computeWrappedLines = useCallback((ctx: CanvasRenderingContext2D, maxWidth: number): WrappedLine[] => {
    const lines: WrappedLine[] = [];
    const spaceWidth = ctx.measureText(" ").width;
    let globalWordCounter = 0;

    for (let vIdx = 0; vIdx < verses.length; vIdx++) {
      const verse = verses[vIdx];
      const timingList = verseLettersMap.get(vIdx) || [];
      const rawWords = verse.text.trim().split(/\s+/);

      let timingPtr = 0;
      const wordLayouts: WordWithLetterBoundaries[] = [];

      for (let wIdx = 0; wIdx < rawWords.length; wIdx++) {
        const w = rawWords[wIdx];
        const graphemes = splitIntoGraphemes(w);
        const lBoundaries: LetterBoundary[] = [];
        let accumulatedPrefix = "";

        let prominentTajweed: TajweedSignDetail = getExactTajweedSign(w[0] || "");

        for (let gIdx = 0; gIdx < graphemes.length; gIdx++) {
          const g = graphemes[gIdx];
          accumulatedPrefix += g;
          const prefixWidth = ctx.measureText(accumulatedPrefix).width;

          const matched = timingList[timingPtr];
          let lStart = 0;
          let lEnd = 0;
          if (matched) {
            lStart = matched.start;
            lEnd = matched.end;
            timingPtr++;
          }

          const nextG = graphemes[gIdx + 1];
          const prevG = graphemes[gIdx - 1];
          const tajweed = getExactTajweedSign(g, nextG, prevG);
          if (tajweed.rule !== "normal" && tajweed.rule !== "silent_wasl") {
            prominentTajweed = tajweed;
          }

          lBoundaries.push({
            char: g,
            start: lStart,
            end: lEnd,
            prefixPixelWidth: prefixWidth,
            tajweed,
          });
        }

        const wStart = lBoundaries.find(l => l.start > 0)?.start || 0;
        const wEnd = [...lBoundaries].reverse().find(l => l.end > 0)?.end || 0;
        const wTotalWidth = ctx.measureText(w).width;

        wordLayouts.push({
          wordIdx: globalWordCounter++,
          wordText: w,
          totalWidth: wTotalWidth,
          start: wStart,
          end: wEnd,
          letters: lBoundaries,
          tajweed: prominentTajweed,
        });
      }

      // Wrap Words into Balanced Lines
      let currentLineWords: WordWithLetterBoundaries[] = [];
      let currentLineWidth = 0;

      for (let wIdx = 0; wIdx < wordLayouts.length; wIdx++) {
        const wl = wordLayouts[wIdx];
        const isLastWord = wIdx === wordLayouts.length - 1;
        const medallionWidth = isLastWord ? ctx.measureText(` ۝${toArabicNumerals(verse.ayah)} `).width : 0;
        const addedWidth = (currentLineWords.length > 0 ? spaceWidth : 0) + wl.totalWidth + (isLastWord ? medallionWidth : 0);

        if (currentLineWidth + addedWidth > maxWidth && currentLineWords.length > 0) {
          const lStart = currentLineWords.find(w => w.start > 0)?.start || 0;
          const lEnd = [...currentLineWords].reverse().find(w => w.end > 0)?.end || 0;
          const lineText = currentLineWords.map(w => w.wordText).join(" ");

          lines.push({
            vIdx,
            ayahNum: verse.ayah,
            isLastLineOfVerse: false,
            words: currentLineWords,
            lineArabicText: lineText,
            totalWidth: currentLineWidth,
            start: lStart,
            end: lEnd,
          });

          currentLineWords = [wl];
          currentLineWidth = wl.totalWidth;
        } else {
          currentLineWords.push(wl);
          currentLineWidth += addedWidth;
        }
      }

      if (currentLineWords.length > 0) {
        const lStart = currentLineWords.find(w => w.start > 0)?.start || 0;
        const lEnd = [...currentLineWords].reverse().find(w => w.end > 0)?.end || 0;
        const lineText = currentLineWords.map(w => w.wordText).join(" ");

        lines.push({
          vIdx,
          ayahNum: verse.ayah,
          isLastLineOfVerse: true,
          words: currentLineWords,
          lineArabicText: lineText,
          totalWidth: currentLineWidth,
          start: lStart,
          end: lEnd,
        });
      }
    }

    cachedLinesRef.current = lines;
    return lines;
  }, [verses, verseLettersMap]);

  // Main High-Precision GPU Render Function
  const renderFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, width, height);

    const isMobile = width < 500;
    const fontSize = isMobile ? 24 : 32;
    const lineHeight = fontSize * 2.2;
    const paddingX = isMobile ? 16 : 28;
    const paddingY = 24;

    ctx.font = `bold ${fontSize}px "Amiri Quran", "Amiri", "Noto Naskh Arabic", serif`;
    ctx.textBaseline = "middle";

    const maxLineWidth = width - paddingX * 2;
    const wrappedLines = computeWrappedLines(ctx, maxLineWidth);
    const spaceWidth = ctx.measureText(" ").width;

    // Apply Psychoacoustic Lead-Bias (-40ms)
    const effectiveTime = currentTime + 0.040;
    let currentY = paddingY + lineHeight / 2;

    for (let i = 0; i < wrappedLines.length; i++) {
      const line = wrappedLines[i];
      line.lineY = currentY;
      const isVerseActive = line.vIdx === activeVerseIdx;
      const isLinePast = line.end > 0 && effectiveTime >= line.end;
      const isLineActive = isVerseActive && effectiveTime >= line.start && effectiveTime < line.end;

      const arabicText = line.lineArabicText;
      const arabicMetrics = ctx.measureText(arabicText);

      let medallionWidth = 0;
      const ayahMedallion = line.isLastLineOfVerse ? ` ۝${toArabicNumerals(line.ayahNum)} ` : "";
      if (line.isLastLineOfVerse) {
        medallionWidth = ctx.measureText(ayahMedallion).width;
      }

      const totalLineWidth = arabicMetrics.width + medallionWidth;
      const lineStartX = Math.min(width - paddingX, (width + totalLineWidth) / 2);
      line.lineStartX = lineStartX;
      const arabicLeftX = lineStartX - arabicMetrics.width;
      const medallionStartX = arabicLeftX;

      // 1. ACTIVE AYAH GLASS CARD BACKDROP
      if (isVerseActive) {
        ctx.save();
        const cardH = lineHeight + 16;
        const cardY = currentY - cardH / 2;
        const cardW = width - paddingX;
        const cardX = paddingX / 2;

        // Radial obsidian-emerald illumination under active verse
        const bgGrad = ctx.createLinearGradient(cardX, cardY, cardX + cardW, cardY + cardH);
        bgGrad.addColorStop(0, "rgba(6, 78, 59, 0.35)");
        bgGrad.addColorStop(0.5, "rgba(10, 18, 36, 0.85)");
        bgGrad.addColorStop(1, "rgba(6, 78, 59, 0.35)");
        ctx.fillStyle = bgGrad;

        ctx.strokeStyle = "rgba(0, 255, 170, 0.45)";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "rgba(0, 255, 170, 0.35)";
        ctx.shadowBlur = 18;

        ctx.beginPath();
        ctx.roundRect(cardX, cardY, cardW, cardH, 16);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      // Calculate Word Bounds for Layout, Hover & Active Word Pill
      let runningWordRightX = lineStartX;
      let activeWordLayout: WordWithLetterBoundaries | null = null;
      let activeWaveHeadX = lineStartX;
      let accumulatedRecitedPixels = 0;
      let activeTajweedColor = "#00ffaa";

      for (let wIdx = 0; wIdx < line.words.length; wIdx++) {
        const w = line.words[wIdx];
        if (wIdx > 0) {
          runningWordRightX -= spaceWidth;
        }

        const wordX = runningWordRightX - w.totalWidth;
        w.renderedX = wordX;
        w.renderedY = currentY - lineHeight / 2;
        w.renderedWidth = w.totalWidth;
        w.renderedHeight = lineHeight;

        const isWordActive = isVerseActive && w.start > 0 && effectiveTime >= w.start && effectiveTime < w.end;
        if (isWordActive) {
          activeWordLayout = w;
          if (enableTajweedColors && w.tajweed?.color) {
            activeTajweedColor = w.tajweed.color;
          }
        }

        if (w.end > 0 && effectiveTime >= w.end) {
          accumulatedRecitedPixels = lineStartX - (runningWordRightX - w.totalWidth);
          activeWaveHeadX = runningWordRightX - w.totalWidth;
        } else if (isWordActive) {
          let prevPrefixWidth = 0;
          for (let lIdx = 0; lIdx < w.letters.length; lIdx++) {
            const l = w.letters[lIdx];
            const curPrefixWidth = l.prefixPixelWidth;
            const letterSpanWidth = curPrefixWidth - prevPrefixWidth;

            if (l.end > 0 && effectiveTime >= l.end) {
              prevPrefixWidth = curPrefixWidth;
              activeWaveHeadX = runningWordRightX - curPrefixWidth;
              accumulatedRecitedPixels = lineStartX - activeWaveHeadX;
            } else if (l.start > 0 && effectiveTime >= l.start && effectiveTime < l.end) {
              const intraFrac = (effectiveTime - l.start) / Math.max(0.005, l.end - l.start);
              const subLetterOffset = prevPrefixWidth + intraFrac * letterSpanWidth;
              activeWaveHeadX = runningWordRightX - subLetterOffset;
              accumulatedRecitedPixels = lineStartX - activeWaveHeadX;
              if (enableTajweedColors && l.tajweed?.color && l.tajweed.rule !== "normal") {
                activeTajweedColor = l.tajweed.color;
              }
              break;
            } else {
              break;
            }
          }
        }

        runningWordRightX -= w.totalWidth;
      }

      // 2. ACTIVE WORD PILL / SPOTLIGHT
      if (activeWordLayout && activeWordLayout.renderedX !== undefined) {
        ctx.save();
        const pillPadX = 7;
        const pillPadY = 5;
        const pillX = activeWordLayout.renderedX - pillPadX;
        const pillY = currentY - (lineHeight / 2) - pillPadY;
        const pillW = activeWordLayout.totalWidth + pillPadX * 2;
        const pillH = lineHeight + pillPadY * 2;

        ctx.fillStyle = "rgba(0, 255, 170, 0.09)";
        ctx.strokeStyle = activeTajweedColor ? `${activeTajweedColor}66` : "rgba(0, 255, 170, 0.4)";
        ctx.lineWidth = 1.2;
        ctx.shadowColor = activeTajweedColor || "rgba(0, 255, 170, 0.35)";
        ctx.shadowBlur = 12;

        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, 9);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      // 3. HOVERED WORD OUTLINE
      if (hoveredWord && hoveredWord.renderedX !== undefined && hoveredWord.wordIdx === line.words.find(w => w.wordIdx === hoveredWord.wordIdx)?.wordIdx) {
        ctx.save();
        const hPadX = 5;
        const hPadY = 3;
        const hX = (hoveredWord.renderedX || 0) - hPadX;
        const hY = currentY - (lineHeight / 2) - hPadY;
        const hW = hoveredWord.totalWidth + hPadX * 2;
        const hH = lineHeight + hPadY * 2;

        ctx.strokeStyle = "rgba(14, 165, 233, 0.6)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.roundRect(hX, hY, hW, hH, 6);
        ctx.stroke();
        ctx.restore();
      }

      // PASS 1: Base Crisp Silver / Pearl Arabic Text
      ctx.save();
      ctx.direction = "rtl";
      ctx.textAlign = "right";
      ctx.fillStyle = isLinePast ? "#00ffaa" : "#e2e8f0";
      if (isLinePast) {
        ctx.shadowColor = "rgba(0, 255, 170, 0.65)";
        ctx.shadowBlur = 8;
      } else {
        ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
        ctx.shadowBlur = 3;
      }
      ctx.fillText(arabicText, lineStartX, currentY);
      ctx.restore();

      // PASS 2: Liquid Lisan Standard - Multi-Layer Kinetic Depth Glow
      if (isLineActive && accumulatedRecitedPixels > 0) {
        const pulse = 1 + 0.08 * Math.sin(performance.now() / 220);

        // Layer 1: Ambient soft diffuse glow
        ctx.save();
        ctx.beginPath();
        const clipW = Math.max(1, accumulatedRecitedPixels + 2);
        ctx.rect(activeWaveHeadX - 2, currentY - lineHeight, clipW + 10, lineHeight * 2);
        ctx.clip();

        ctx.direction = "rtl";
        ctx.textAlign = "right";
        ctx.fillStyle = activeTajweedColor || "#00ffaa";
        ctx.shadowColor = activeTajweedColor ? `${activeTajweedColor}99` : "rgba(0, 255, 170, 0.6)";
        ctx.shadowBlur = Math.round(20 * pulse);
        ctx.fillText(arabicText, lineStartX, currentY);

        // Layer 2: Core luminous text fill
        ctx.shadowColor = activeTajweedColor || "#00ffaa";
        ctx.shadowBlur = 8;
        ctx.fillText(arabicText, lineStartX, currentY);
        ctx.restore();

        // Layer 3: Wavefront Laser Crest Bloom right on active letter playhead
        ctx.save();
        ctx.beginPath();
        const laserW = Math.min(22, accumulatedRecitedPixels);
        ctx.rect(activeWaveHeadX - 3, currentY - lineHeight, laserW + 6, lineHeight * 2);
        ctx.clip();

        ctx.direction = "rtl";
        ctx.textAlign = "right";
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = Math.round(18 * pulse);
        ctx.fillText(arabicText, lineStartX, currentY);
        ctx.restore();

        // Layer 4: Fine Laser Cursor Beam at activeWaveHeadX
        ctx.save();
        const beamGrad = ctx.createLinearGradient(0, currentY - lineHeight / 2, 0, currentY + lineHeight / 2);
        beamGrad.addColorStop(0, "rgba(56, 189, 248, 0)");
        beamGrad.addColorStop(0.5, "#ffffff");
        beamGrad.addColorStop(1, "rgba(56, 189, 248, 0)");
        ctx.fillStyle = beamGrad;
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = Math.round(10 * pulse);
        ctx.fillRect(activeWaveHeadX - 1, currentY - lineHeight / 2, 2.5, lineHeight);
        ctx.restore();
      }

      // PASS 3: Draw Sacred Golden Ayah Medallion
      if (line.isLastLineOfVerse) {
        ctx.save();
        ctx.direction = "rtl";
        ctx.textAlign = "right";
        ctx.fillStyle = "#fbbf24"; // Pure Celestial Gold
        ctx.shadowColor = "rgba(251, 191, 36, 0.75)";
        ctx.shadowBlur = 12;
        ctx.fillText(ayahMedallion, medallionStartX, currentY);
        ctx.restore();
      }

      currentY += lineHeight + 14;
    }
  }, [verses, currentTime, activeVerseIdx, enableTajweedColors, hoveredWord, computeWrappedLines]);

  // Handle High-DPI Canvas Resizing
  const handleResize = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const w = rect.width || 360;

    const isMobile = w < 500;
    const fontSize = isMobile ? 24 : 32;
    const lineHeight = fontSize * 2.2;
    const paddingX = isMobile ? 16 : 28;

    const dummyCanvas = document.createElement("canvas");
    const dummyCtx = dummyCanvas.getContext("2d");
    if (!dummyCtx) return;
    dummyCtx.font = `bold ${fontSize}px "Amiri Quran", "Amiri", "Noto Naskh Arabic", serif`;

    const maxLineWidth = w - paddingX * 2;
    const lines = computeWrappedLines(dummyCtx, maxLineWidth);
    const totalHeight = lines.length * (lineHeight + 14) + 70;

    canvas.width = w * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${totalHeight}px`;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
    renderFrame();
  }, [computeWrappedLines, renderFrame]);

  // RequestAnimationFrame 60-120 FPS Loop
  useEffect(() => {
    handleResize();
    window.addEventListener("resize", handleResize);

    let animationFrameId: number;
    const loop = () => {
      renderFrame();
      if (isPlaying) {
        animationFrameId = requestAnimationFrame(loop);
      }
    };

    if (isPlaying) {
      animationFrameId = requestAnimationFrame(loop);
    } else {
      renderFrame();
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, handleResize, renderFrame]);

  // Smooth Auto-Scroll to Active Verse / Line in Mushaf Mode
  useEffect(() => {
    if (!autoScroll || !isPlaying) return;
    const lines = cachedLinesRef.current;
    const activeLine = lines.find(l => l.vIdx === activeVerseIdx && l.lineY !== undefined);
    if (activeLine && activeLine.lineY !== undefined && canvasRef.current) {
      const canvasRect = canvasRef.current.getBoundingClientRect();
      const targetScrollY = window.scrollY + canvasRect.top + activeLine.lineY - (window.innerHeight / 2);
      window.scrollTo({
        top: Math.max(0, targetScrollY),
        behavior: "smooth",
      });
    }
  }, [activeVerseIdx, autoScroll, isPlaying]);

  // Mouse Move Detection for Precision Word Hover & Tooltip
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    let foundWord: WordWithLetterBoundaries | null = null;
    const lines = cachedLinesRef.current;

    for (const line of lines) {
      if (line.lineY && Math.abs(mouseY - line.lineY) < 30) {
        for (const w of line.words) {
          if (
            w.renderedX !== undefined &&
            w.renderedWidth !== undefined &&
            mouseX >= w.renderedX &&
            mouseX <= w.renderedX + w.renderedWidth
          ) {
            foundWord = w;
            break;
          }
        }
      }
      if (foundWord) break;
    }

    if (foundWord !== hoveredWord) {
      setHoveredWord(foundWord);
      if (onWordHover) {
        onWordHover(foundWord ? foundWord.wordText : null, foundWord ? foundWord.tajweed : undefined);
      }
    }
  };

  const handleMouseLeave = () => {
    setHoveredWord(null);
    if (onWordHover) onWordHover(null);
  };

  // Click to Seek directly to Clicked Word or Line
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const lines = cachedLinesRef.current;
    for (const line of lines) {
      if (line.lineY && Math.abs(clickY - line.lineY) < 30) {
        // Check if a specific word was clicked
        for (const w of line.words) {
          if (
            w.renderedX !== undefined &&
            w.renderedWidth !== undefined &&
            clickX >= w.renderedX &&
            clickX <= w.renderedX + w.renderedWidth
          ) {
            if (w.start > 0) {
              onSeek(w.start);
              return;
            }
          }
        }
        // Fallback: Seek to line start
        if (line.start > 0) {
          onSeek(line.start);
          return;
        }
      }
    }
  };

  return (
    <div ref={containerRef} className="fluid-mushaf-canvas-wrapper" dir="rtl">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="fluid-mushaf-canvas"
        style={{ cursor: hoveredWord ? "pointer" : "default" }}
        title="Sacred Fluid Mushaf Canvas (Precision Sub-Letter Karaoke & Tajweed Highlighting)"
      />
    </div>
  );
};
