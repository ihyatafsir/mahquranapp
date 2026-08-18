import { WaveformStudio } from "./components/WaveformStudio";
import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useLetterSync } from './hooks/useLetterSync';
import ThreeBackground from './components/ThreeBackground';
import type { LetterTiming, Verse, TimedWord, TimedLetter } from './types/quran';
import './index.css';

// Available Reciters (Sheikh Mohammad Ahmad Hassan as default primary)
const RECITERS = [
  {
    id: "minshawi_mujawwad",
    name: "Sheikh Mohamed Siddiq Al-Minshawi",
    shortName: "Al-Minshawi (Mujawwad)",
    description: "Egyptian Master Reciter • Classical Tahqeeq Style",
  },
  {
    id: 'mah',
    name: 'Sheikh Mohammad Ahmad Hassan',
    shortName: 'Mohammad Ahmad Hassan (MAH)',
    description: 'Acoustic Alignment & Tajweed Guided Physics',
  },
  {
    id: 'abdul_basit',
    name: 'Sheikh AbdulBaset AbdulSamad',
    shortName: 'Abdul Basit (Mujawwad)',
    description: 'Egyptian Master Reciter',
  },
];

// All 24 Surahs available with full audio + timing for MAH
const MAH_SURAHS = [
  { number: 1, name: 'Al-Fatiha', arabicName: 'الفاتحة', meaning: 'The Opening', versesCount: 7 },
  { number: 2, name: 'Al-Baqarah', arabicName: 'البقرة', meaning: 'The Cow', versesCount: 286 },
  { number: 18, name: 'Al-Kahf', arabicName: 'الكهف', meaning: 'The Cave', versesCount: 110 },
  { number: 36, name: 'Ya-Sin', arabicName: 'يس', meaning: 'Ya-Sin', versesCount: 83 },
  { number: 47, name: 'Muhammad', arabicName: 'محمد', meaning: 'Muhammad', versesCount: 38 },
  { number: 53, name: 'An-Najm', arabicName: 'النجم', meaning: 'The Star', versesCount: 62 },
  { number: 55, name: 'Ar-Rahman', arabicName: 'الرحمن', meaning: 'The Beneficent', versesCount: 78 },
  { number: 56, name: 'Al-Waqi\'ah', arabicName: 'الواقعة', meaning: 'The Inevitable', versesCount: 96 },
  { number: 67, name: 'Al-Mulk', arabicName: 'الملك', meaning: 'The Sovereignty', versesCount: 30 },
  { number: 71, name: 'Nuh', arabicName: 'نوح', meaning: 'Noah', versesCount: 28 },
  { number: 75, name: 'Al-Qiyamah', arabicName: 'القيامة', meaning: 'The Resurrection', versesCount: 40 },
  { number: 80, name: 'Abasa', arabicName: 'عبس', meaning: 'He Frowned', versesCount: 42 },
  { number: 82, name: 'Al-Infitar', arabicName: 'الانفطار', meaning: 'The Cleaving', versesCount: 19 },
  { number: 85, name: 'Al-Buruj', arabicName: 'البروج', meaning: 'The Mansions of the Stars', versesCount: 22 },
  { number: 87, name: 'Al-A\'la', arabicName: 'الأعلى', meaning: 'The Most High', versesCount: 19 },
  { number: 89, name: 'Al-Fajr', arabicName: 'الفجر', meaning: 'The Dawn', versesCount: 30 },
  { number: 90, name: 'Al-Balad', arabicName: 'البلد', meaning: 'The City', versesCount: 20 },
  { number: 91, name: 'Ash-Shams', arabicName: 'الشمس', meaning: 'The Sun', versesCount: 15 },
  { number: 92, name: 'Al-Layl', arabicName: 'الليل', meaning: 'The Night', versesCount: 21 },
  { number: 93, name: 'Ad-Duha', arabicName: 'الضحى', meaning: 'The Morning Hours', versesCount: 11 },
  { number: 109, name: 'Al-Kafirun', arabicName: 'الكافرون', meaning: 'The Disbelievers', versesCount: 6 },
  { number: 112, name: 'Al-Ikhlas', arabicName: 'الإخلاص', meaning: 'The Sincerity', versesCount: 4 },
  { number: 113, name: 'Al-Falaq', arabicName: 'الفلق', meaning: 'The Daybreak', versesCount: 5 },
  { number: 114, name: 'An-Nas', arabicName: 'الناس', meaning: 'Mankind', versesCount: 6 },
];

const ABDUL_BASIT_SURAHS = [
  { number: 1, name: 'Al-Fatiha', arabicName: 'الفاتحة', meaning: 'The Opening', versesCount: 7 },
  { number: 2, name: 'Al-Baqarah', arabicName: 'البقرة', meaning: 'The Cow', versesCount: 286 },
  { number: 3, name: 'Al-Imran', arabicName: 'آل عمران', meaning: 'Family of Imran', versesCount: 200 },
  { number: 4, name: 'An-Nisa', arabicName: 'النساء', meaning: 'The Women', versesCount: 176 },
  { number: 5, name: 'Al-Ma\'idah', arabicName: 'المائدة', meaning: 'The Table Spread', versesCount: 120 },
];

const MINSHAWI_SURAHS = [
  { number: 1, name: "Al-Fatiha", arabicName: "الفاتحة", meaning: "The Opening", versesCount: 7 },
  { number: 112, name: "Al-Ikhlas", arabicName: "الإخلاص", meaning: "The Sincerity", versesCount: 4 },
];

const SURAHS_BY_RECITER: Record<string, typeof MAH_SURAHS> = {
  minshawi_mujawwad: MINSHAWI_SURAHS,
  mah: MAH_SURAHS,
  abdul_basit: ABDUL_BASIT_SURAHS,
};

// Decompose an Arabic string into base letters + diacritics
function splitArabicIntoLetters(text: string): string[] {
  const DIACRITICS = new Set([
    "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650", "\u0651", "\u0652",
    "\u0653", "\u0654", "\u0655", "\u0656", "\u0657", "\u0658", "\u065C", "\u065D",
    "\u065E", "\u065F", "\u0670", "\u06E1", "\u06DF", "\u06E0", "\u06E2", "\u06E3"
  ]);
  const chunks: string[] = [];
  let curr = "";
  for (const char of text) {
    if (DIACRITICS.has(char)) {
      curr += char;
    } else {
      if (curr) chunks.push(curr);
      curr = char;
    }
  }
  if (curr) chunks.push(curr);
  return chunks;
}

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
  const [selectedReciter, setSelectedReciter] = useState('mah');
  const [selectedSurah, setSelectedSurah] = useState(36); // Default to Surah 36 (Ya-Sin) or 1
  const [verses, setVerses] = useState<Verse[]>([]);
  const [letterTiming, setLetterTiming] = useState<LetterTiming[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStudioOpen, setIsStudioOpen] = useState<boolean>(false);

  // UI View Preferences
  const [showWordCards, setShowWordCards] = useState(true);
  const [showTranslation, setShowTranslation] = useState(true);
  const [showThreeBg, setShowThreeBg] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  const audioRef = useRef<HTMLAudioElement>(null);
  const verseRefs = useRef<Map<number, HTMLElement>>(new Map());
  const activeWordRef = useRef<HTMLSpanElement | null>(null);

  const syncState = useLetterSync(audioRef, letterTiming);

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
        const versesRes = await fetch('/data/verses_v4.json');
        if (!versesRes.ok) throw new Error('Could not load verse definitions');
        const allVerses = await versesRes.json();
        const loadedVerses: Verse[] = allVerses[selectedSurah.toString()] || [];

        if (isCancelled) return;
        setVerses(loadedVerses);

        // Load timing from reciter-specific path
        const timingPath =
          selectedReciter === "mah"
            ? `/data/letter_timing_${selectedSurah}.json`
            : `/data/${selectedReciter}/letter_timing_${selectedSurah}.json`;

        const timingRes = await fetch(timingPath);
        if (timingRes.ok) {
          const rawTiming: LetterTiming[] = await timingRes.json();
          if (!isCancelled) setLetterTiming(rawTiming);
        } else {
          if (!isCancelled) setLetterTiming([]);
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

  // Seek and Play handlers
  const handleSeek = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, time);
      if (audioRef.current.paused) {
        audioRef.current.play().catch(() => {});
      }
    }
  };

  const handleWordClick = useCallback((word: TimedWord) => {
    handleSeek(word.start);
  }, []);

  const handleVerseClick = (verseIdx: number) => {
    const wordsForVerse = verseTimedWords.get(verseIdx) || [];
    if (wordsForVerse.length > 0) {
      handleSeek(wordsForVerse[0].start);
    }
  };

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      audioRef.current.play().catch(e => console.warn('Playback error:', e));
    } else {
      audioRef.current.pause();
    }
  };

  const handleSkip = (seconds: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(
      0,
      Math.min(audioRef.current.currentTime + seconds, audioRef.current.duration || 0)
    );
  };

  const handlePrevVerse = () => {
    const targetIdx = Math.max(0, syncState.currentVerseIdx - 1);
    handleVerseClick(targetIdx);
  };

  const handleNextVerse = () => {
    const targetIdx = Math.min(verses.length - 1, syncState.currentVerseIdx + 1);
    handleVerseClick(targetIdx);
  };

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
        {/* Header */}
        <header className="header-card">
          <button className="control-btn" style={{ borderColor: "#00ff88", color: "#00ff88" }} onClick={() => setIsStudioOpen(true)}>🎙️ Tajweed Studio</button>
        <div className="header-badge">
            <span className="live-indicator" />
            HIGH-PRECISION LETTER-BY-LETTER RECITATION KARAOKE
          </div>
          <h1 className="header-title">
            <span className="glow-text">القرآن الكريم</span>
            <span className="title-sub">MAH Letter Timing Precision</span>
          </h1>
          <p className="reciter-subtitle">
            Reciter:{' '}
            <strong>{RECITERS.find(r => r.id === selectedReciter)?.name}</strong>
          </p>
        </header>

        {/* Navigation & Selection Controls */}
        <section className="controls-card">
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

          {/* Quick Toggles */}
          <div className="toggles-row">
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

        {/* Quran Text & Karaoke Display */}
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
          ) : (
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
                    onClick={() => handleVerseClick(verseIdx)}
                  >
                    {/* Verse Header */}
                    <div className="verse-meta-bar">
                      <span className="ayah-chip" title={`Surah ${selectedSurah}, Ayah ${verse.ayah}`}>
                        {selectedSurah}:{verse.ayah}
                      </span>
                      <span className="ayah-hint">Click anywhere in Ayah to jump</span>
                    </div>

                    {/* Arabic Text with Letter Karaoke Glowing */}
                    <div className="arabic-karaoke-block" dir="rtl">
                      {wordsInVerse.length > 0 ? (
                        wordsInVerse.map(word => {
                          const isWordActive = word.globalWordIdx === syncState.currentWordIdx;
                          const isWordPast =
                            syncState.currentWordIdx >= 0 && word.globalWordIdx < syncState.currentWordIdx;

                          return (
                            <span
                              key={word.globalWordIdx}
                              ref={isWordActive ? activeWordRef : undefined}
                              className={`word-span ${isWordActive ? 'word-active' : ''} ${
                                isWordPast ? 'word-past' : ''
                              }`}
                              onClick={e => {
                                e.stopPropagation();
                                handleWordClick(word);
                              }}
                              title={`Word #${word.globalWordIdx + 1} (${word.start.toFixed(2)}s - ${word.end.toFixed(2)}s)`}
                            >
                              {word.letters.map((letter, letterIdxInWord) => {
                                const isLetterActive =
                                  isWordActive &&
                                  currentLetter &&
                                  currentLetter.wordIdx === word.globalWordIdx &&
                                  (typeof (currentLetter as any).charIdxInWord !== "undefined"
                                    ? (currentLetter as any).charIdxInWord === letterIdxInWord
                                    : currentLetter.char === letter.char);

                                const isLetterPast =
                                  (syncState.currentWordIdx >= 0 && word.globalWordIdx < syncState.currentWordIdx) ||
                                  (isWordActive && currentLetter && (
                                    typeof (currentLetter as any).charIdxInWord !== "undefined"
                                      ? letterIdxInWord < (currentLetter as any).charIdxInWord
                                      : false
                                  ));

                                return (
                                  <span
                                    key={letterIdxInWord}
                                    className={`letter-span ${isLetterActive ? "letter-active" : ""} ${
                                      isLetterPast ? "letter-past" : ""
                                    }`}
                                  >
                                    {letter.char}
                                  </span>
                                );
                              })}
                            </span>
                          );
                        })
                      ) : (
                        <span className="verse-fallback-text">{verse.text}</span>
                      )}
                      <span className="ayah-end-symbol"> ﴿{verse.ayah}﴾ </span>
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

                          return (
                            <div
                              key={w.id || wIdx}
                              className={`word-card-chip ${isActive ? 'active' : ''} ${
                                isPast ? 'past' : ''
                              }`}
                              onClick={e => {
                                e.stopPropagation();
                                if (matchedWord) handleWordClick(matchedWord);
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
          )}
        </main>
      </div>
    </div>
  );
}
