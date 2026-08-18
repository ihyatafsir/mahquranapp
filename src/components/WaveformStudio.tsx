import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { LetterTiming } from '../types/quran';

interface WaveformStudioProps {
  isOpen: boolean;
  onClose: () => void;
  audioSrc: string;
  letterTiming: LetterTiming[];
  reciterName: string;
  surahName: string;
  onSaveTiming: (newTiming: LetterTiming[]) => void;
}

export const WaveformStudio: React.FC<WaveformStudioProps> = ({
  isOpen,
  onClose,
  audioSrc,
  letterTiming,
    reciterName,
  surahName,
  onSaveTiming,
}: WaveformStudioProps) => {
  const [timing, setTiming] = useState<LetterTiming[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [zoom, setZoom] = useState<number>(1.0);
  const [statusMsg, setStatusMsg] = useState<string>('');

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Sync prop changes
  useEffect(() => {
    if (letterTiming && letterTiming.length > 0) {
      setTiming(JSON.parse(JSON.stringify(letterTiming)));
      setSelectedIdx(0);
    }
  }, [letterTiming]);

  // Audio play/pause & time update
  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      audio.play().catch(() => {});
      setIsPlaying(true);
    } else {
      audio.pause();
      setIsPlaying(false);
    }
  };

  // Play a specific letter slice
  const auditionLetter = useCallback((letter: LetterTiming) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = letter.start;
    audio.play().catch(() => {});
    setIsPlaying(true);

    const checkEnd = () => {
      if (!audioRef.current) return;
      if (audioRef.current.currentTime >= letter.end) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        requestAnimationFrame(checkEnd);
      }
    };
    requestAnimationFrame(checkEnd);
  }, []);

  // Nudge timing of selected letter
  const nudgeTiming = (deltaStart: number, deltaEnd: number) => {
    if (selectedIdx < 0 || selectedIdx >= timing.length) return;

    setTiming(prev => {
      const copy = [...prev];
      const target = { ...copy[selectedIdx] };
      target.start = Math.max(0, Number((target.start + deltaStart).toFixed(3)));
      target.end = Math.max(target.start + 0.05, Number((target.end + deltaEnd).toFixed(3)));
      target.duration = Number((target.end - target.start).toFixed(3));
      copy[selectedIdx] = target;
      return copy;
    });

    setStatusMsg(`Nudged #${selectedIdx + 1} ("${timing[selectedIdx]?.char}"): ${deltaStart !== 0 ? (deltaStart > 0 ? '+' : '') + deltaStart*1000 + 'ms start' : ''} ${deltaEnd !== 0 ? (deltaEnd > 0 ? '+' : '') + deltaEnd*1000 + 'ms end' : ''}`);
  };

  // Save changes
  const handleSave = () => {
    onSaveTiming(timing);
    setStatusMsg('Saved and applied timing to live player!');
    setTimeout(() => setStatusMsg(''), 3000);
  };

  // Draw Waveform & Letter Banners
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isOpen) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, '#0a101d');
    bgGrad.addColorStop(1, '#050b14');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    if (duration <= 0 || timing.length === 0) return;

    // Time window based on current zoom and selected letter
    const selLetter = timing[selectedIdx] || timing[0];
    const windowDur = (duration / zoom);
    const centerT = selLetter ? (selLetter.start + selLetter.end) / 2 : currentTime;
    const minT = Math.max(0, centerT - windowDur / 2);
    const maxT = Math.min(duration, minT + windowDur);

    const timeToX = (t: number) => ((t - minT) / (maxT - minT)) * width;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
    ctx.lineWidth = 1;
    for (let t = Math.floor(minT); t <= Math.ceil(maxT); t += 0.5) {
      const x = timeToX(t);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
      ctx.font = '10px monospace';
      ctx.fillText(`${t.toFixed(1)}s`, x + 4, 14);
    }

    // Draw Letter Boxes
    const boxY = height - 90;
    const boxH = 50;

    timing.forEach((l, idx) => {
      if (l.end < minT || l.start > maxT) return;

      const x1 = timeToX(l.start);
      const x2 = timeToX(l.end);
      const w = Math.max(2, x2 - x1);
      const isSelected = idx === selectedIdx;

      // Fill
      ctx.fillStyle = isSelected 
        ? 'rgba(0, 255, 136, 0.35)' 
        : (idx % 2 === 0 ? 'rgba(56, 189, 248, 0.15)' : 'rgba(124, 58, 237, 0.15)');
      ctx.fillRect(x1, boxY, w, boxH);

      // Border
      ctx.strokeStyle = isSelected ? '#00ff88' : 'rgba(56, 189, 248, 0.4)';
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.strokeRect(x1, boxY, w, boxH);

      // Letter text
      ctx.fillStyle = isSelected ? '#ffffff' : '#cbd5e1';
      ctx.font = isSelected ? 'bold 16px "Amiri", serif' : '14px "Amiri", serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(l.char, x1 + w / 2, boxY + boxH / 2);
    });

    // Draw Playhead
    const playheadX = timeToX(currentTime);
    if (playheadX >= 0 && playheadX <= width) {
      ctx.strokeStyle = '#ff007f';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
    }
  }, [isOpen, timing, selectedIdx, currentTime, duration, zoom]);

  if (!isOpen) return null;

  const currentLetter = timing[selectedIdx];
  const letterDur = currentLetter ? (currentLetter.duration ?? (currentLetter.end - currentLetter.start)) : 0;

  return (
    <div className="waveform-studio-overlay">
      <div className="waveform-studio-modal">
        {/* Header */}
        <div className="studio-header">
          <div className="studio-title-group">
            <span className="studio-badge">TAJWEED WAVE STUDIO</span>
            <h2>{surahName} • {reciterName}</h2>
          </div>
          <button className="studio-close-btn" onClick={onClose} title="Close Studio">✕</button>
        </div>

        {/* Studio Controls */}
        <div className="studio-toolbar">
          <button className="studio-btn primary" onClick={togglePlay}>
            {isPlaying ? '❚❚ Pause' : '▶ Play'}
          </button>
          <button 
            className="studio-btn" 
            onClick={() => currentLetter && auditionLetter(currentLetter)}
            disabled={!currentLetter}
          >
            🔊 Audition Active Letter
          </button>

          <div className="zoom-group">
            <label>Zoom:</label>
            <input 
              type="range" 
              min="1.0" 
              max="8.0" 
              step="0.5" 
              value={zoom} 
              onChange={e => setZoom(Number(e.target.value))} 
            />
            <span>{zoom.toFixed(1)}x</span>
          </div>

          <div className="studio-actions">
            <button className="studio-btn success" onClick={handleSave}>💾 Save & Apply</button>
          </div>
        </div>

        {/* Waveform Canvas */}
        <div className="canvas-wrapper">
          <canvas 
            ref={canvasRef} 
            width={960} 
            height={200} 
            className="studio-canvas"
          />
        </div>

        {/* Micro-Adjustment Editor Card */}
        {currentLetter && (
          <div className="letter-inspector-card">
            <div className="inspector-left">
              <div className="active-char-box">{currentLetter.char}</div>
              <div className="char-meta">
                <div className="meta-row">
                  <span className="label">Index:</span>
                  <span className="val">#{selectedIdx + 1} of {timing.length}</span>
                </div>
                <div className="meta-row">
                  <span className="label">Time:</span>
                  <span className="val">{currentLetter.start.toFixed(3)}s – {currentLetter.end.toFixed(3)}s</span>
                </div>
                <div className="meta-row">
                  <span className="label">Duration:</span>
                  <span className="val">{(letterDur * 1000).toFixed(0)} ms</span>
                </div>
              </div>
            </div>

            <div className="inspector-controls">
              <div className="nudge-cluster">
                <span className="nudge-title">Start Point:</span>
                <div className="btn-group">
                  <button className="nudge-btn" onClick={() => nudgeTiming(-0.02, 0)}>-20ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(-0.01, 0)}>-10ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(0.01, 0)}>+10ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(0.02, 0)}>+20ms</button>
                </div>
              </div>

              <div className="nudge-cluster">
                <span className="nudge-title">End Point:</span>
                <div className="btn-group">
                  <button className="nudge-btn" onClick={() => nudgeTiming(0, -0.02)}>-20ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(0, -0.01)}>-10ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(0, 0.01)}>+10ms</button>
                  <button className="nudge-btn" onClick={() => nudgeTiming(0, 0.02)}>+20ms</button>
                </div>
              </div>

              <div className="nav-cluster">
                <button 
                  className="studio-btn" 
                  disabled={selectedIdx <= 0}
                  onClick={() => setSelectedIdx(prev => Math.max(0, prev - 1))}
                >
                  ◀ Prev
                </button>
                <button 
                  className="studio-btn" 
                  disabled={selectedIdx >= timing.length - 1}
                  onClick={() => setSelectedIdx(prev => Math.min(timing.length - 1, prev + 1))}
                >
                  Next ▶
                </button>
              </div>
            </div>
          </div>
        )}

        {statusMsg && <div className="studio-status-banner">{statusMsg}</div>}

        <audio 
          ref={audioRef} 
          src={audioSrc} 
          onTimeUpdate={() => audioRef.current && setCurrentTime(audioRef.current.currentTime)}
          onLoadedMetadata={() => audioRef.current && setDuration(audioRef.current.duration)}
        />
      </div>
    </div>
  );
};
