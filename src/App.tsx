import { useState, useRef, useEffect, useMemo } from 'react';
import { useLetterSync } from './hooks/useLetterSync';
import type { LetterTiming, Verse } from './types/quran';
import './index.css';

// Reciters
const RECITERS = [
  { id: 'mah', name: 'Sheikh Muhammad Ahmad Hassan', shortName: 'MAH' },
  { id: 'abdul_basit', name: 'Sheikh AbdulBaset AbdulSamad (Mujawwad)', shortName: 'Abdul Basit' },
];

// Surahs with audio + letter timing (per reciter)
const SURAHS_BY_RECITER: Record<string, { number: number; name: string; arabicName: string }[]> = {
  mah: [
    { number: 1, name: 'Al-Fatiha', arabicName: 'الفاتحة' },
    { number: 2, name: 'Al-Baqarah', arabicName: 'البقرة' },
    { number: 18, name: 'Al-Kahf', arabicName: 'الكهف' },
    { number: 36, name: 'Ya-Sin', arabicName: 'يس' },
    { number: 55, name: 'Ar-Rahman', arabicName: 'الرحمن' },
    { number: 56, name: 'Al-Waqiah', arabicName: 'الواقعة' },
    { number: 67, name: 'Al-Mulk', arabicName: 'الملك' },
  ],
  abdul_basit: [
    { number: 1, name: 'Al-Fatiha', arabicName: 'الفاتحة' },
    { number: 2, name: 'Al-Baqarah', arabicName: 'البقرة' },
    { number: 3, name: 'Al-Imran', arabicName: 'آل عمران' },
    { number: 4, name: 'An-Nisa', arabicName: 'النساء' },
    { number: 5, name: 'Al-Maidah', arabicName: 'المائدة' },
  ],
};

// Normalize timing to seconds
function normalizeTimingToSeconds(timing: LetterTiming[]): LetterTiming[] {
  if (timing.length === 0) return timing;
  const firstStart = timing[0].start;
  const isMilliseconds = firstStart > 100;
  if (isMilliseconds) {
    return timing.map(t => ({
      ...t,
      start: t.start / 1000,
      end: t.end / 1000,
      duration: t.duration / 1000
    }));
  }
  return timing;
}

// Group letters by wordIdx
interface TimedWord {
  wordIdx: number;
  letters: { char: string; globalIdx: number; start: number; end: number }[];
  text: string;
  start: number;
  end: number;
}

function groupLettersIntoWords(timing: LetterTiming[]): TimedWord[] {
  const words: TimedWord[] = [];
  let currentWord: TimedWord | null = null;

  timing.forEach((letter, globalIdx) => {
    if (!currentWord || letter.wordIdx !== currentWord.wordIdx) {
      if (currentWord) words.push(currentWord);
      currentWord = {
        wordIdx: letter.wordIdx,
        letters: [],
        text: '',
        start: letter.start,
        end: letter.end
      };
    }
    currentWord.letters.push({
      char: letter.char,
      globalIdx,
      start: letter.start,
      end: letter.end
    });
    currentWord.text += letter.char;
    currentWord.end = letter.end;
  });

  if (currentWord) words.push(currentWord);
  return words;
}

// Distribute timed words to verses
function distributeToVerses(timedWords: TimedWord[], verses: Verse[]): Map<number, TimedWord[]> {
  const verseWords = new Map<number, TimedWord[]>();
  let wordOffset = 0;

  verses.forEach((verse, verseIdx) => {
    const verseWordCount = verse.text.split(' ').length;
    const wordsForVerse = timedWords.slice(wordOffset, wordOffset + verseWordCount);
    verseWords.set(verseIdx, wordsForVerse);
    wordOffset += verseWordCount;
  });

  return verseWords;
}

function App() {
  const [selectedReciter, setSelectedReciter] = useState('abdul_basit');
  const [selectedSurah, setSelectedSurah] = useState(1);
  const [verses, setVerses] = useState<Verse[]>([]);
  const [letterTiming, setLetterTiming] = useState<LetterTiming[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const syncState = useLetterSync(audioRef, letterTiming);

  const availableSurahs = SURAHS_BY_RECITER[selectedReciter] || [];
  const surahInfo = availableSurahs.find(s => s.number === selectedSurah);

  // Group timing data into words
  const timedWords = useMemo(() => groupLettersIntoWords(letterTiming), [letterTiming]);

  // Distribute to verses
  const verseTimedWords = useMemo(() => distributeToVerses(timedWords, verses), [timedWords, verses]);

  // Load data
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        const versesRes = await fetch('/data/verses_v4.json');
        const allVerses = await versesRes.json();
        setVerses(allVerses[selectedSurah.toString()] || []);

        // Load timing from reciter-specific path
        const timingPath = selectedReciter === 'mah'
          ? `/data/letter_timing_${selectedSurah}.json`
          : `/data/${selectedReciter}/letter_timing_${selectedSurah}.json`;

        const timingRes = await fetch(timingPath);
        if (timingRes.ok) {
          let rawTiming: LetterTiming[] = await timingRes.json();
          rawTiming = normalizeTimingToSeconds(rawTiming);
          setLetterTiming(rawTiming);
        } else {
          setLetterTiming([]);
        }
      } catch (err) {
        setError('Failed to load data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedSurah, selectedReciter]);

  const currentLetter = syncState.currentLetterIdx >= 0 ? letterTiming[syncState.currentLetterIdx] : null;

  return (
    <div className="app-container">
      <header className="header">
        <h1>💿 MAH Quran</h1>
        <p>Letter-by-letter recitation with {RECITERS.find(r => r.id === selectedReciter)?.name}</p>
      </header>

      <div className="controls">
        <select
          value={selectedReciter}
          onChange={(e) => {
            setSelectedReciter(e.target.value);
            setSelectedSurah(SURAHS_BY_RECITER[e.target.value]?.[0]?.number || 1);
          }}
          className="surah-select"
          style={{ marginRight: '10px' }}
        >
          {RECITERS.map(reciter => (
            <option key={reciter.id} value={reciter.id}>
              {reciter.shortName}
            </option>
          ))}
        </select>
        <select
          value={selectedSurah}
          onChange={(e) => setSelectedSurah(Number(e.target.value))}
          className="surah-select"
        >
          {availableSurahs.map(surah => (
            <option key={surah.number} value={surah.number}>
              {surah.number}. {surah.name} ({surah.arabicName})
            </option>
          ))}
        </select>

        <label className="debug-toggle">
          <input
            type="checkbox"
            checked={showDebug}
            onChange={(e) => setShowDebug(e.target.checked)}
          />
          Debug
        </label>
      </div>

      <div className="audio-player">
        <audio
          ref={audioRef}
          controls
          src={selectedReciter === 'mah'
            ? `/audio/surah_${selectedSurah.toString().padStart(3, '0')}.mp3`
            : `/audio/${selectedReciter}/surah_${selectedSurah.toString().padStart(3, '0')}.mp3`
          }
        />
      </div>

      <div className="version-tag" style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        color: 'var(--text-secondary)',
        fontSize: '0.8rem',
        opacity: 0.7
      }}>
        v1.1 (Al-Baqarah)
      </div>

      {showDebug && (
        <div className="debug-panel">
          <span><b>Time:</b> {syncState.currentTime.toFixed(2)}s</span>
          <span><b>Letter:</b> {syncState.currentLetterIdx}/{letterTiming.length}</span>
          <span><b>Word:</b> {syncState.currentWordIdx}/{timedWords.length}</span>
          {currentLetter && <span><b>Char:</b> "{currentLetter.char}"</span>}
        </div>
      )}

      {loading ? (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading Surah {surahInfo?.name}...</p>
        </div>
      ) : error ? (
        <div className="loading">
          <p className="error">{error}</p>
        </div>
      ) : (
        <div className="verses-container">
          {verses.map((verse, verseIdx) => {
            const wordsInVerse = verseTimedWords.get(verseIdx) || [];

            return (
              <div key={verse.ayah} className="verse">
                <div className="verse-header">
                  <span className="verse-number">{verse.ayah}</span>
                </div>

                {/* Arabic Text with letter highlighting */}
                <div className="arabic-text">
                  {wordsInVerse.map((word) => {
                    const isWordActive = word.wordIdx === syncState.currentWordIdx;
                    const isWordPast = word.wordIdx < syncState.currentWordIdx;

                    return (
                      <span
                        key={word.wordIdx}
                        className={`word ${isWordActive ? 'word-active' : ''} ${isWordPast ? 'word-past' : ''}`}
                      >
                        {word.letters.map((letter) => {
                          const isLetterActive = letter.globalIdx === syncState.currentLetterIdx;
                          const isLetterPast = letter.globalIdx < syncState.currentLetterIdx;

                          return (
                            <span
                              key={letter.globalIdx}
                              className={`letter ${isLetterActive ? 'active' : ''} ${isLetterPast ? 'past' : ''}`}
                            >
                              {letter.char}
                            </span>
                          );
                        })}
                      </span>
                    );
                  })}
                </div>

                {/* Word Cards with Transliteration */}
                {verse.words && verse.words.length > 0 && (
                  <div className="word-cards-row">
                    {verse.words.map((word, wIdx) => {
                      // Map to global word index
                      const globalWordIdx = wordsInVerse[wIdx]?.wordIdx ?? -1;
                      const isActive = globalWordIdx === syncState.currentWordIdx;

                      return (
                        <div key={wIdx} className={`word-card ${isActive ? 'active' : ''}`}>
                          <span className="word-arabic">{word.arabic}</span>
                          <span className="word-translit">{word.translit}</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Translation */}
                <div className="translation">
                  {verse.translation}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default App;
