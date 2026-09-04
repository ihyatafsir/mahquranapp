import React, { useRef, useEffect, useState, useCallback } from "react";
import type { TimedWord, TimedLetter, LetterTiming } from "../types/quran";

interface AudioWaveformSyncCanvasProps {
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  activeWord?: TimedWord | null;
  activeLetter?: LetterTiming | TimedLetter | null;
  letterTiming?: LetterTiming[];
  reciterId: string;
  surahNumber: number;
  onSeek: (time: number) => void;
}

export const AudioWaveformSyncCanvas: React.FC<AudioWaveformSyncCanvasProps> = ({
  currentTime,
  duration,
  isPlaying,
  activeWord,
  letterTiming = [],
  reciterId,
  surahNumber,
  onSeek,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [peaks, setPeaks] = useState<number[]>([]);
  const [waveformDuration, setWaveformDuration] = useState<number>(duration || 41.5);
  const animationFrameRef = useRef<number | null>(null);
  const [canvasWidth, setCanvasWidth] = useState<number>(950);

  // Responsive resize observer
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const w = Math.max(320, rect.width - 24);
        setCanvasWidth(w);
      }
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Fetch real waveform data
  useEffect(() => {
    let isCancelled = false;
    const fetchWaveform = async () => {
      try {
        const url = `/data/waveforms/${reciterId}_surah_${surahNumber}.json`;
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          if (!isCancelled) {
            setPeaks(data.peaks || []);
            setWaveformDuration(data.duration || duration);
          }
        } else {
          // Fallback synthetic harmonic peaks
          const syntheticPeaks = Array.from({ length: 400 }, (_, i) => {
            const t = i / 10;
            return Math.max(0.06, Math.abs(Math.sin(t * 1.5) * Math.cos(t * 0.7) * 0.75 + Math.random() * 0.15));
          });
          if (!isCancelled) setPeaks(syntheticPeaks);
        }
      } catch (err) {
        console.warn("Waveform load error:", err);
      }
    };
    fetchWaveform();
    return () => {
      isCancelled = true;
    };
  }, [reciterId, surahNumber, duration]);

  // High-Precision 60-120 FPS Canvas Draw Loop in Authentic Right-to-Left (RTL) Flow
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvasWidth;
    const height = 90;

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const centerY = height / 2 + 10;

    // Obsidian Glass Waveform Backdrop
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, "rgba(10, 15, 29, 0.95)");
    bgGrad.addColorStop(1, "rgba(5, 8, 18, 0.98)");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    if (peaks.length === 0) {
      ctx.restore();
      return;
    }

    const totalDur = waveformDuration || 41.5;
    const playheadFraction = Math.max(0, Math.min(1, currentTime / totalDur));
    // In RTL: t=0 is at width (far right), t=totalDur is at 0 (far left)
    const playheadX = width - (playheadFraction * width);

    // 1. Draw Audio Waveform Bars (RTL Flow)
    const barCount = Math.min(peaks.length, Math.floor(width / 3.2));
    const barStep = width / barCount;
    const peakStep = peaks.length / barCount;

    for (let i = 0; i < barCount; i++) {
      const peakIdx = Math.floor(i * peakStep);
      const amp = peaks[peakIdx] || 0.05;
      const barHeight = Math.max(3, amp * (height * 0.65));
      // RTL bar placement: i=0 is on the far right
      const x = width - ((i + 1) * barStep);

      // In RTL, recited audio is to the RIGHT of the playhead (x >= playheadX)
      const isPast = x >= playheadX;

      if (isPast) {
        ctx.fillStyle = "#00ffaa";
        ctx.shadowColor = "rgba(0, 255, 170, 0.5)";
        ctx.shadowBlur = 4;
      } else {
        ctx.fillStyle = "#334155";
        ctx.shadowBlur = 0;
      }

      ctx.fillRect(x, centerY - barHeight / 2, Math.max(1.5, barStep - 1), barHeight);
    }
    ctx.shadowBlur = 0;

    // 2. Draw Active Word Highlight Box on Waveform (RTL)
    if (activeWord && activeWord.end > activeWord.start) {
      const wordStartX = width - ((activeWord.end / totalDur) * width);
      const wordEndX = width - ((activeWord.start / totalDur) * width);
      const wordW = Math.max(4, wordEndX - wordStartX);

      ctx.fillStyle = "rgba(0, 240, 255, 0.15)";
      ctx.fillRect(wordStartX, 0, wordW, height);

      ctx.strokeStyle = "rgba(0, 240, 255, 0.6)";
      ctx.lineWidth = 1;
      ctx.strokeRect(wordStartX, 0, wordW, height);
    }

    // 3. Draw Waveplace Vocalisation Tags in RTL
    ctx.font = 'bold 12px "Amiri Quran", "Amiri", sans-serif';
    ctx.textAlign = "center";

    letterTiming.forEach(l => {
      const pTime = (l as any).peakTime || (l.start + l.end) / 2;
      const tagX = width - ((pTime / totalDur) * width);

      const isLetterActive = currentTime >= l.start && currentTime < l.end;
      const isLetterPast = currentTime >= l.end;

      if (isLetterActive) {
        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tagX, 20);
        ctx.lineTo(tagX, centerY);
        ctx.stroke();

        ctx.fillStyle = "#00ffaa";
        ctx.shadowColor = "#00ffaa";
        ctx.shadowBlur = 10;
        ctx.fillText(l.char, tagX, 16);
        ctx.shadowBlur = 0;
      } else if (isLetterPast) {
        ctx.fillStyle = "rgba(0, 255, 170, 0.75)";
        ctx.fillText(l.char, tagX, 16);
      } else {
        ctx.fillStyle = "rgba(100, 116, 139, 0.55)";
        ctx.fillText(l.char, tagX, 16);
      }
    });

    // 4. Draw Playhead Laser Beam (RTL Sweep)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#00f0ff";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();

    // Playhead glowing diamond/circle head
    ctx.fillStyle = "#00f0ff";
    ctx.beginPath();
    ctx.arc(playheadX, centerY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.restore();
  }, [canvasWidth, peaks, currentTime, waveformDuration, activeWord, letterTiming]);

  // Animation frame loop
  useEffect(() => {
    let active = true;
    const renderLoop = () => {
      if (!active) return;
      drawWaveform();
      if (isPlaying) {
        animationFrameRef.current = requestAnimationFrame(renderLoop);
      }
    };
    renderLoop();
    return () => {
      active = false;
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [drawWaveform, isPlaying]);

  // RTL Click-to-Seek: Clicking on far right seeks to 0.0s
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, (rect.width - clickX) / rect.width));
    const targetTime = fraction * (waveformDuration || 41.5);
    onSeek(targetTime);
  };

  return (
    <div ref={containerRef} className="waveform-sync-container" dir="rtl">
      <div className="waveform-header">
        <div className="waveform-label">
          <span className="waveform-live-dot" />
          <span>WAVEPLACE VOCALISATION ALIGNER • Right-to-Left (RTL) Acoustic Flow</span>
        </div>
        <div className="waveform-time-display" dir="ltr">
          {currentTime.toFixed(2)}s / {waveformDuration.toFixed(2)}s
        </div>
      </div>
      <canvas
        ref={canvasRef}
        className="waveform-sync-canvas"
        onClick={handleCanvasClick}
        title="Waveplace (RTL): Click anywhere from Right (Start) to Left (End) to seek instantly"
      />
    </div>
  );
};
