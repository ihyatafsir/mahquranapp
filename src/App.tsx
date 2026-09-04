import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { FluidMushafCanvas } from './components/FluidMushafCanvas';
import { WaveformStudio } from "./components/WaveformStudio";
import { MakhrajVisualizer } from "./components/MakhrajVisualizer";
import { BiomechanicalVocalHUD } from "./components/BiomechanicalVocalHUD";
import { AudioWaveformSyncCanvas } from "./components/AudioWaveformSyncCanvas";
import { WordWaveformPill } from "./components/WordWaveformPill";
import { MobileTajweedInspectorHUD } from "./components/MobileTajweedInspectorHUD";
import ThreeBackground from './components/ThreeBackground';
import { useLetterSync } from './hooks/useLetterSync';
import { RECITERS } from './constants/reciters';
import { SURAHS_BY_RECITER, MAH_SURAHS } from './constants/surahs';
import { toArabicNumerals, splitIntoGraphemes as splitArabicIntoLetters } from './utils/arabic';
import { fetchWithCache, preloadSurahTiming } from './utils/quranCache';
import type { LetterTiming, Verse, TimedWord, TimedLetter } from './types/quran';
import './index.css';

// Group canonical Quran words with letter timing
function groupLettersIntoWords(timing: LetterTiming[], verses: Verse[]): TimedWord[] {
  if (!verses || verses.length === 0) return [];

  const words: TimedWord[] = [];
  let globalWIdx = 0;

  verses.forEach((verse, vIdx) => {
    const wordList = verse.words && verse.words.length > 0
      ? verse.words
      : verse.text.trim().split(/\s+/).map(w => ({ arabic: w }));

    wordList.forEach(w => {
      const chunks = splitArabicIntoLetters(w.arabic);
      const timedLetters: TimedLetter[] = chunks.map((c, lIdx) => ({
        char: c,
        globalIdx: -1,
        charIdxInWord: lIdx,
        start: 0,
        end: 0,
      }));

      // Find timing matching this word
      const matchingLetterTimings = timing.filter(t => t.wordIdx === globalWIdx);
      let wStart = 0;
      let wEnd = 0;
      if (matchingLetterTimings.length > 0) {
        wStart = matchingLetterTimings[0].start;
        wEnd = matchingLetterTimings[matchingLetterTimings.length - 1].end;
      }

      words.push({
        globalWordIdx: globalWIdx,
        verseIdx: vIdx,
        ayah: verse.ayah,
        letters: timedLetters,
        text: w.arabic,
        start: wStart,
        end: wEnd,
        arabic: w.arabic,
        translit: (w as any).translit,
        root: (w as any).root,
      });

      globalWIdx++;
    });
  });

  return words;
}

// Distribute TimedWords into verse buckets
function distributeToVerses(timedWords: TimedWord[], verses: Verse[]): Map<number, TimedWord[]> {
  const verseMap = new Map<number, TimedWord[]>();
  verses.forEach((_, vIdx) => {
    verseMap.set(vIdx, []);
  });

  timedWords.forEach(word => {
    const vIdx = word.verseIdx;
    if (verseMap.has(vIdx)) {
      verseMap.get(vIdx)!.push(word);
    } else {
      verseMap.set(vIdx, [word]);
    }
  });

  return verseMap;
}

// Format seconds into MM:SS
function formatTime(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function App() {
  const [selectedReciter, setSelectedReciter] = useState('abdul_basit_murattal');
  const [selectedSurah, setSelectedSurah] = useState(1); // Default to Surah 36 (Ya-Sin) or 1
  const [verses, setVerses] = useState<Verse[]>([]);
  const [letterTiming, setLetterTiming] = useState<LetterTiming[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStudioOpen, setIsStudioOpen] = useState<boolean>(false);

  // UI View Preferences
  const [viewMode, setViewMode] = useState<'mushaf' | 'flow'>('mushaf');
  const [auditioningWordIdx, setAuditioningWordIdx] = useState<number | null>(null);
  const stopAtTimeRef = useRef<number | null>(null);

  const [showWordCards, setShowWordCards] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);
  const [showThreeBg, setShowThreeBg] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [showSettingsDrawer, setShowSettingsDrawer] = useState(false);
  const [showMakhraj, setShowMakhraj] = useState(false);
  const [enableTajweedColors, setEnableTajweedColors] = useState(true);
  const [showTajweedLegend, setShowTajweedLegend] = useState(false);
  const [hoveredWordText, setHoveredWordText] = useState<string | null>(null);
  const [hoveredTajweed, setHoveredTajweed] = useState<any>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  const audioRef = useRef<HTMLAudioElement>(null);
  const verseRefs = useRef<Map<number, HTMLElement>>(new Map());
  // const activeWordRef = useRef<HTMLSpanElement | null>(null);

  const handleStopReached = useCallback(() => {
    setAuditioningWordIdx(null);
  }, []);

  const syncState = useLetterSync(audioRef, letterTiming, 0, stopAtTimeRef, handleStopReached);

  const availableSurahs = SURAHS_BY_RECITER[selectedReciter] || MAH_SURAHS;
  const currentSurahInfo = availableSurahs.find(s => s.number === selectedSurah) || availableSurahs[0];

  // Group timing data into words
  const timedWords = useMemo(
    () => groupLettersIntoWords(letterTiming, verses),
    [letterTiming, verses]
  );

  // Group words into verses
  const verseTimedWords = useMemo(
    () => distributeToVerses(timedWords, verses),
    [timedWords, verses]
  );

  // Load Quranic data and letter timing
  useEffect(() => {
    let isCancelled = false;

    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        const allVerses = await fetchWithCache<Record<string, Verse[]>>('/data/verses_v4.json');
        const loadedVerses: Verse[] = allVerses[selectedSurah.toString()] || [];

        if (isCancelled) return;
        setVerses(loadedVerses);

        // Load timing from reciter-specific path via multi-layer cache
        const timingPath =
          selectedReciter === "mah"
            ? `/data/letter_timing_${selectedSurah}.json`
            : `/data/${selectedReciter}/letter_timing_${selectedSurah}.json`;

        try {
          const rawTiming = await fetchWithCache<LetterTiming[]>(timingPath);
          if (!isCancelled) setLetterTiming(rawTiming);
        } catch {
          if (!isCancelled) setLetterTiming([]);
        }

        // Predictive pre-warming of adjacent surahs in background
        preloadSurahTiming(selectedSurah + 1, selectedReciter);
        if (selectedSurah > 1) {
          preloadSurahTiming(selectedSurah - 1, selectedReciter);
        }
      } catch (err: unknown) {
        if (!isCancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load recitation data');
          console.error(err);
        }
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    loadData();
    return () => {
      isCancelled = true;
    };
  }, [selectedSurah, selectedReciter]);

  // Audio Playback Speed Control
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed]);

  // Smooth Auto-scroll to active verse
  useEffect(() => {
    if (!autoScroll || !syncState.isPlaying) return;

    const currentVerseEl = verseRefs.current.get(syncState.currentVerseIdx);
    if (currentVerseEl) {
      currentVerseEl.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [syncState.currentVerseIdx, autoScroll, syncState.isPlaying]);

  // Play only a single word's audio (WhisperX precision alignment boundary)
  const playSingleWord = useCallback((word: TimedWord) => {
    if (!audioRef.current) return;
    if (word.start === 0 && word.end === 0) return;

    // Add +35ms psychoacoustic padding so trailing acoustic releases aren't clipped
    const stopTime = word.end > word.start ? word.end + 0.035 : word.start + 0.6;
    stopAtTimeRef.current = stopTime;
    setAuditioningWordIdx(word.globalWordIdx);

    audioRef.current.currentTime = Math.max(0, word.start);
    audioRef.current.play().catch(e => console.warn('Word audition playback error:', e));
  }, []);

  // When clicking a word in Flow Mode: play ONLY that word audio!
  const handleWordClickInFlow = useCallback((word: TimedWord) => {
    playSingleWord(word);
  }, [playSingleWord]);

  // Seek and Play handlers (clears isolated word audition bounds for continuous recitation)
  const handleSeek = (time: number) => {
    stopAtTimeRef.current = null;
    setAuditioningWordIdx(null);
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, time);
      if (audioRef.current.paused) {
        audioRef.current.play().catch(() => {});
      }
    }
  };

  const handleVerseClick = (verseIdx: number) => {
    stopAtTimeRef.current = null;
    setAuditioningWordIdx(null);
    const wordsForVerse = verseTimedWords.get(verseIdx) || [];
    if (wordsForVerse.length > 0) {
      handleSeek(wordsForVerse[0].start);
    }
  };

  const handlePlayPause = useCallback(() => {
    stopAtTimeRef.current = null;
    setAuditioningWordIdx(null);
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      audioRef.current.play().catch(e => console.warn('Playback error:', e));
    } else {
      audioRef.current.pause();
    }
  }, []);

  const handleSkip = useCallback((seconds: number) => {
    stopAtTimeRef.current = null;
    setAuditioningWordIdx(null);
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(
      0,
      Math.min(audioRef.current.currentTime + seconds, audioRef.current.duration || 0)
    );
  }, []);

  const handlePrevVerse = useCallback(() => {
    const targetIdx = Math.max(0, syncState.currentVerseIdx - 1);
    handleVerseClick(targetIdx);
  }, [syncState.currentVerseIdx, handleVerseClick]);

  const handleNextVerse = useCallback(() => {
    const targetIdx = Math.min(verses.length - 1, syncState.currentVerseIdx + 1);
    handleVerseClick(targetIdx);
  }, [verses.length, syncState.currentVerseIdx, handleVerseClick]);

  // Global Keyboard Shortcuts (Space: Play/Pause, Arrows: Seek & Ayah)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        handlePlayPause();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        handleSkip(-5);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        handleSkip(5);
      } else if (e.code === 'ArrowUp') {
        e.preventDefault();
        handlePrevVerse();
      } else if (e.code === 'ArrowDown') {
        e.preventDefault();
        handleNextVerse();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePlayPause, handleSkip, handlePrevVerse, handleNextVerse]);

  const currentLetter = syncState.currentLetterIdx >= 0 ? letterTiming[syncState.currentLetterIdx] : null;
  const currentWord = timedWords.find(w => w.globalWordIdx === syncState.currentWordIdx);

  // Audio source URL
  const audioSrc = useMemo(() => {
    const padNum = selectedSurah.toString().padStart(3, "0");
    return selectedReciter === "mah"
      ? `/audio/surah_${padNum}.mp3`
      : `/audio/${selectedReciter}/surah_${padNum}.mp3`;
  }, [selectedReciter, selectedSurah]);

  return (
    <div className="app-layout">
      {/* Dynamic Three.js Ambient Particle Background */}
      {showThreeBg && <ThreeBackground isPlaying={syncState.isPlaying} />}

      <div className="app-container">
        {/* Luxury Sacred Top Navigation Bar */}
        <nav className="luxury-top-nav">
          <div className="nav-brand">
            <span className="brand-dot" />
            <span className="brand-title">LIQUID LISAN</span>
            <span className="brand-badge">TajweedSST v2.0</span>
          </div>

          <div className="nav-actions">
            <button
              className={`nav-action-btn ${isStudioOpen ? 'active' : ''}`}
              onClick={() => setIsStudioOpen(true)}
              title="Open Waveform & Alignment Studio"
            >
              🎙️ Studio
            </button>
            <button
              className={`nav-action-btn ${showMakhraj ? 'active' : ''}`}
              onClick={() => setShowMakhraj(!showMakhraj)}
              title="Toggle Vocal Tract Biomechanics & Makhraj"
            >
              👄 Vocal Tract
            </button>
            <button
              className={`nav-action-btn ${showThreeBg ? 'active' : ''}`}
              onClick={() => setShowThreeBg(!showThreeBg)}
              title="Toggle 3D Particle Universe"
            >
              🌌 3D Space
            </button>
            <button
              className={`nav-action-btn ${showSettingsDrawer ? 'active' : ''}`}
              onClick={() => setShowSettingsDrawer(!showSettingsDrawer)}
              title="Toggle Display Settings"
            >
              ⚙️ Settings
            </button>
          </div>
        </nav>

        {/* Centerpiece Header Card */}
        <header className="header-card">
          <div className="header-badge">
            <span className="live-indicator" />
            HIGH-PRECISION PHONEME & LETTER KARAOKE
          </div>
          <h1 className="header-title">
            <span className="glow-text">القرآن الكريم</span>
            <span className="title-sub">{currentSurahInfo.arabicName} • {currentSurahInfo.name} ({currentSurahInfo.meaning})</span>
          </h1>
          <p className="reciter-subtitle">
            Reciter: <strong>{RECITERS.find(r => r.id === selectedReciter)?.name}</strong>
          </p>
        </header>

        {/* Navigation & Selection Controls */}
        <section className="controls-card">
          {/* View Mode Switcher: Sacred Mushaf Canvas vs Interactive Flow Mode */}
          <div className="view-mode-selector-row">
            <button
              type="button"
              className={`view-mode-tab-btn ${viewMode === 'mushaf' ? 'active' : ''}`}
              onClick={() => {
                stopAtTimeRef.current = null;
                setAuditioningWordIdx(null);
                setViewMode('mushaf');
              }}
            >
              <span className="tab-icon">📖</span>
              <span className="tab-label">Mushaf Canvas</span>
              <span className="tab-badge">120 FPS</span>
            </button>
            <button
              type="button"
              className={`view-mode-tab-btn ${viewMode === 'flow' ? 'active' : ''}`}
              onClick={() => {
                stopAtTimeRef.current = null;
                setAuditioningWordIdx(null);
                setViewMode('flow');
              }}
            >
              <span className="tab-icon">🌊</span>
              <span className="tab-label">Interactive Flow Mode</span>
              <span className="tab-badge">Word Audio</span>
            </button>
          </div>

          <div className="select-row">
            <div className="select-group">
              <label htmlFor="reciter-select">Reciter</label>
              <select
                id="reciter-select"
                value={selectedReciter}
                onChange={e => {
                  setSelectedReciter(e.target.value);
                  setSelectedSurah(SURAHS_BY_RECITER[e.target.value]?.[0]?.number || 1);
                }}
                className="custom-select"
              >
                {RECITERS.map(reciter => (
                  <option key={reciter.id} value={reciter.id}>
                    {reciter.shortName}
                  </option>
                ))}
              </select>
            </div>

            <div className="select-group">
              <label htmlFor="surah-select">Surah</label>
              <select
                id="surah-select"
                value={selectedSurah}
                onChange={e => setSelectedSurah(Number(e.target.value))}
                className="custom-select"
              >
                {availableSurahs.map(surah => (
                  <option key={surah.number} value={surah.number}>
                    {surah.number}. {surah.name} ({surah.arabicName}) - {surah.versesCount} Ayahs
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Collapsible Mobile-First Quick Settings */}
          <div className="settings-drawer-toggle-row">
            <button
              className="settings-toggle-btn"
              onClick={() => setShowSettingsDrawer(!showSettingsDrawer)}
            >
              ⚙️ {showSettingsDrawer ? "Hide Display Settings" : "Display & Visual Settings"}
            </button>
          </div>

          {showSettingsDrawer && (
            <div className="toggles-row">
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={enableTajweedColors}
                  onChange={e => setEnableTajweedColors(e.target.checked)}
                />
                <span>🎨 Tajweed Color Rules</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showTajweedLegend}
                  onChange={e => setShowTajweedLegend(e.target.checked)}
                />
                <span>📖 Tajweed Legend</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showWordCards}
                  onChange={e => setShowWordCards(e.target.checked)}
                />
                <span>Word Cards & Roots</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showMakhraj}
                  onChange={e => setShowMakhraj(e.target.checked)}
                />
                <span>👄 Makhraj & Lips</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showTranslation}
                  onChange={e => setShowTranslation(e.target.checked)}
                />
                <span>Translation</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={e => setAutoScroll(e.target.checked)}
                />
                <span>Auto-Scroll</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showThreeBg}
                  onChange={e => setShowThreeBg(e.target.checked)}
                />
                <span>3D Universe</span>
              </label>
              <label className="pill-toggle">
                <input
                  type="checkbox"
                  checked={showDebug}
                  onChange={e => setShowDebug(e.target.checked)}
                />
                <span>Debug HUD</span>
              </label>
            </div>
          )}

          {/* Collapsible Tajweed Color Legend Guide */}
          {showTajweedLegend && (
            <div className="tajweed-legend-card" dir="rtl">
              <div className="legend-title">قواعد وألوان التجويد المعيارية (Standard Mushaf Rules):</div>
              <div className="legend-grid">
                <div className="legend-item" style={{ borderColor: "#f43f5e" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#f43f5e" }} />
                  <span className="legend-name">مد لازم (6 حركات)</span>
                  <span className="legend-symbol">ٓ (ضَّآلِّينَ)</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#fb923c" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#fb923c" }} />
                  <span className="legend-name">مد واجب/جائز (4-5 حركات)</span>
                  <span className="legend-symbol">ٓ (جَآءَ)</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#10b981" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#10b981" }} />
                  <span className="legend-name">غنة الحرف المشدد (2 ح)</span>
                  <span className="legend-symbol">نّ / مّ</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#34d399" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#34d399" }} />
                  <span className="legend-name">إقلاب</span>
                  <span className="legend-symbol">ۢ (مِنۢ بَعْدِ)</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#06b6d4" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#06b6d4" }} />
                  <span className="legend-name">قلقلة (قطب جد ساكن)</span>
                  <span className="legend-symbol">ْ (يَخْرُجُ)</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#818cf8" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#818cf8" }} />
                  <span className="legend-name">تفخيم واستعلاء</span>
                  <span className="legend-symbol">خص ضغط قظ</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#f59e0b" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#f59e0b" }} />
                  <span className="legend-name">مد طبيعي (2 حركة)</span>
                  <span className="legend-symbol">ٰ (الألف الخنجرية)</span>
                </div>
                <div className="legend-item" style={{ borderColor: "#94a3b8" }}>
                  <span className="legend-color-dot" style={{ backgroundColor: "#94a3b8" }} />
                  <span className="legend-name">همزة وصل / لا يُنطق</span>
                  <span className="legend-symbol">ٱ / ۟</span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Master Glassmorphic Audio Player Bar */}
        <section className="audio-player-card">
          <audio ref={audioRef} src={audioSrc} preload="auto" />

          <div className="player-top-row">
            <div className="surah-now-playing">
              <span className="surah-badge">{currentSurahInfo.number}</span>
              <div className="surah-meta">
                <span className="surah-title-arabic">{currentSurahInfo.arabicName}</span>
                <span className="surah-title-latin">{currentSurahInfo.name} • {currentSurahInfo.meaning}</span>
              </div>
            </div>

            <div className="verse-counter">
              Ayah <span className="counter-accent">{syncState.currentVerseIdx + 1}</span> of {verses.length}
            </div>
          </div>

          {/* Time Scrubber */}
          <div className="scrubber-container">
            <span className="time-text">{formatTime(syncState.currentTime)}</span>
            <input
              type="range"
              min="0"
              max={syncState.duration || 100}
              step="0.05"
              value={syncState.currentTime}
              onChange={e => handleSeek(Number(e.target.value))}
              className="scrub-bar"
            />
            <span className="time-text">{formatTime(syncState.duration)}</span>
          </div>

          {/* Action Buttons */}
          <div className="player-controls">
            <button
              onClick={handlePrevVerse}
              className="control-btn"
              title="Previous Ayah"
              aria-label="Previous Ayah"
            >
              ⏮ Ayah
            </button>
            <button
              onClick={() => handleSkip(-5)}
              className="control-btn"
              title="Replay 5 seconds"
              aria-label="Replay 5 seconds"
            >
              ↺ -5s
            </button>
            <button
              onClick={handlePlayPause}
              className={`play-main-btn ${syncState.isPlaying ? 'is-playing' : ''}`}
              aria-label={syncState.isPlaying ? 'Pause' : 'Play'}
            >
              {syncState.isPlaying ? '❚❚' : '▶'}
            </button>
            <button
              onClick={() => handleSkip(5)}
              className="control-btn"
              title="Skip 5 seconds"
              aria-label="Skip 5 seconds"
            >
              +5s ↻
            </button>
            <button
              onClick={handleNextVerse}
              className="control-btn"
              title="Next Ayah"
              aria-label="Next Ayah"
            >
              Ayah ⏭
            </button>

            {/* Speed Selector */}
            <div className="speed-selector">
              {[0.75, 1.0, 1.25, 1.5].map(speed => (
                <button
                  key={speed}
                  onClick={() => setPlaybackSpeed(speed)}
                  className={`speed-btn ${playbackSpeed === speed ? 'active' : ''}`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Tajweed Wave Studio Modal */}
      <WaveformStudio
        isOpen={isStudioOpen}
        onClose={() => setIsStudioOpen(false)}
        audioSrc={audioSrc}
        letterTiming={letterTiming}
                reciterName={RECITERS.find(r => r.id === selectedReciter)?.shortName || ""}
        surahName={currentSurahInfo?.name || ""}
        onSaveTiming={(newTiming) => {
          setLetterTiming(newTiming);
          try {
            localStorage.setItem(`custom_timing_${selectedReciter}_${selectedSurah}`, JSON.stringify(newTiming));
          } catch(e) {}
        }}
      />

        {/* Live Debug HUD */}
        {showDebug && (
          <aside className="debug-hud-card">
            <div className="debug-item">
              <span className="debug-label">Audio Time:</span>
              <span className="debug-val">{syncState.currentTime.toFixed(3)}s / {syncState.duration.toFixed(3)}s</span>
            </div>
            <div className="debug-item">
              <span className="debug-label">Active Letter:</span>
              <span className="debug-val">
                {syncState.currentLetterIdx >= 0 ? `#${syncState.currentLetterIdx} ("${currentLetter?.char}")` : 'Idle'}
              </span>
            </div>
            <div className="debug-item">
              <span className="debug-label">Active Word:</span>
              <span className="debug-val">
                {syncState.currentWordIdx >= 0 ? `#${syncState.currentWordIdx} ("${currentWord?.text || ''}")` : 'Idle'}
              </span>
            </div>
            <div className="debug-item">
              <span className="debug-label">Active Ayah:</span>
              <span className="debug-val">{syncState.currentVerseIdx + 1} / {verses.length}</span>
            </div>
          </aside>
        )}

        {/* RTL Waveform & Live Tajweed Inspector (Immediate Top Focus) */}
        <AudioWaveformSyncCanvas
          currentTime={syncState.currentTime}
          duration={audioRef.current?.duration || 41.5}
          isPlaying={syncState.isPlaying}
          activeWord={currentWord}
          activeLetter={currentLetter}
          letterTiming={letterTiming}
          reciterId={selectedReciter}
          surahNumber={selectedSurah}
          onSeek={handleSeek}
        />

        <MobileTajweedInspectorHUD
          activeWord={currentWord || null}
          activeLetter={currentLetter || null}
          isPlaying={syncState.isPlaying}
          hoveredTajweed={hoveredTajweed}
          hoveredWordText={hoveredWordText}
        />

        <main className="quran-content-card">
          {loading ? (
            <div className="loading-state">
              <div className="cyber-spinner" />
              <p>Loading {currentSurahInfo.name} ({currentSurahInfo.arabicName})...</p>
            </div>
          ) : error ? (
            <div className="error-state">
              <span className="error-icon">⚠️</span>
              <p>{error}</p>
            </div>
          ) : viewMode === 'mushaf' ? (
            <FluidMushafCanvas
              verses={verses}
              letterTiming={letterTiming}
              currentTime={syncState.currentTime}
              isPlaying={syncState.isPlaying}
              activeVerseIdx={syncState.currentVerseIdx}
              enableTajweedColors={enableTajweedColors}
              autoScroll={autoScroll}
              onSeek={handleSeek}
              onWordHover={(wordText, tajweed) => {
                setHoveredWordText(wordText);
                setHoveredTajweed(tajweed || null);
              }}
            />
          ) : (
            <div className="flow-mode-container">
              {/* Flow Mode Helper Banner */}
              <div className="flow-mode-helper-banner">
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span className="banner-icon">🎧</span>
                  <span><strong>Flow Mode:</strong> Tap any word to play <em>only that word</em> (WhisperX precision). Tap ۝ Ayah or ▶ for continuous recitation.</span>
                </div>
                <span className="banner-tag">WhisperX Aligned</span>
              </div>

              <div className="verses-list">
                {verses.map((verse, verseIdx) => {
                  const wordsInVerse = verseTimedWords.get(verseIdx) || [];
                  const isVerseActive = verseIdx === syncState.currentVerseIdx;

                  return (
                    <article
                      key={verse.ayah}
                      ref={el => {
                        if (el) verseRefs.current.set(verseIdx, el);
                        else verseRefs.current.delete(verseIdx);
                      }}
                      className={`verse-item ${isVerseActive ? 'verse-active' : ''}`}
                    >
                      {/* Arabic Text with Letter Karaoke Glowing & Precision Word Audition */}
                      <div className="arabic-karaoke-block" dir="rtl">
                        {wordsInVerse.length > 0 ? (
                          wordsInVerse.map(word => {
                            const isWordActive = word.globalWordIdx === syncState.currentWordIdx;
                            const isWordPast =
                              syncState.currentWordIdx >= 0 && word.globalWordIdx < syncState.currentWordIdx;
                            const isAuditioning = auditioningWordIdx === word.globalWordIdx;

                            return (
                              <WordWaveformPill
                                key={word.globalWordIdx}
                                word={word}
                                isWordActive={isWordActive}
                                isWordPast={isWordPast}
                                isAuditioning={isAuditioning}
                                enableTajweedColors={enableTajweedColors}
                                currentTime={syncState.currentTime}
                                activeLetter={currentLetter as any}
                                onWordClick={handleWordClickInFlow}
                                onWordHover={(wordText, tajweed) => {
                                  setHoveredWordText(wordText);
                                  setHoveredTajweed(tajweed || null);
                                }}
                              />
                            );
                          })
                        ) : (
                          <span className="verse-fallback-text">{verse.text}</span>
                        )}
                        <span
                          className="inline-ayah-marker"
                          title={`Ayah ${verse.ayah} • Tap to recite Ayah`}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleVerseClick(verseIdx);
                          }}
                          style={{ cursor: 'pointer' }}
                        >
                          {' '}۝{toArabicNumerals(verse.ayah)}{' '}
                        </span>
                      </div>

                      {/* Word Cards Row with Transliteration & Roots */}
                      {showWordCards && verse.words && verse.words.length > 0 && (
                        <div className="word-cards-flow" dir="rtl">
                          {verse.words.map((w, wIdx) => {
                            const matchedWord = wordsInVerse[wIdx];
                            const isActive =
                              matchedWord && matchedWord.globalWordIdx === syncState.currentWordIdx;
                            const isPast =
                              matchedWord &&
                              syncState.currentWordIdx >= 0 &&
                              matchedWord.globalWordIdx < syncState.currentWordIdx;
                            const isAuditioning = matchedWord && auditioningWordIdx === matchedWord.globalWordIdx;

                            return (
                              <div
                                key={w.id || wIdx}
                                className={`word-card-chip ${isActive ? 'active' : ''} ${
                                  isPast ? 'past' : ''
                                } ${isAuditioning ? 'auditioning' : ''}`}
                                title={`Play word: ${w.arabic}`}
                                onClick={e => {
                                  e.stopPropagation();
                                  if (matchedWord) handleWordClickInFlow(matchedWord);
                                }}
                              >
                                <span className="card-arabic">{w.arabic}</span>
                                {w.translit && <span className="card-translit">{w.translit}</span>}
                                {w.root && <span className="card-root">Root: {w.root}</span>}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Translation */}
                      {showTranslation && verse.translation && (
                        <div className="translation-text">
                          {verse.translation}
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </div>
          )}
        </main>

        {/* Biomechanical Vocal Apparatus & Visualizer (Positioned Below Quran Reader) */}
        {showMakhraj && <MakhrajVisualizer currentChar={currentLetter ? currentLetter.char : null} isPlaying={syncState.isPlaying} />}
        {showMakhraj && <BiomechanicalVocalHUD currentLetter={currentLetter} isPlaying={syncState.isPlaying} reciterId={selectedReciter} />}
      </div>
    </div>
  );
}
