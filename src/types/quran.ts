// TypeScript types for Quran data structures

export interface Word {
    id: number;
    arabic: string;
    translit: string;
    root: string;
}

export interface Verse {
    ayah: number;
    text: string;
    translation: string;
    words: Word[];
    hasIhya?: boolean;
}

export interface SurahData {
    [surahNumber: string]: Verse[];
}

// Letter timing from audio alignment
export interface LetterTiming {
    charIdx: number;
    char: string;
    start: number;
    end: number;
    duration?: number;
    wordIdx: number;
    verseIdx?: number;
    ayah?: number;
    idx?: number;
}

// Verse timing
export interface VerseTiming {
    ayah: number;
    start: number;
    end: number;
}

// Available surah info
export interface SurahInfo {
    number: number;
    name: string;
    arabicName: string;
    meaning?: string;
    hasAudio: boolean;
    hasLetterTiming: boolean;
}

// Audio sync state
export interface SyncState {
    currentTime: number;
    duration: number;
    currentLetterIdx: number;
    currentWordIdx: number;
    currentVerseIdx: number;
    letterProgress: number; // 0.0 to 1.0 intra-letter progression
    isPlaying: boolean;
}

// Grouped word representation for karaoke
export interface TimedLetter {
    char: string;
    globalIdx: number;
    start: number;
    end: number;
}

export interface TimedWord {
    globalWordIdx: number;
    verseIdx: number;
    ayah: number;
    letters: TimedLetter[];
    text: string;
    start: number;
    end: number;
    arabic?: string;
    translit?: string;
    root?: string;
}
