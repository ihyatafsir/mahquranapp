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
    hasIhya: boolean;
}

export interface SurahData {
    [surahNumber: string]: Verse[];
}

// Letter timing from MAH audio
export interface LetterTiming {
    charIdx: number;
    char: string;
    start: number;
    end: number;
    duration: number;
    wordIdx: number;
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
    hasAudio: boolean;
    hasLetterTiming: boolean;
}

// Audio sync state
export interface SyncState {
    currentTime: number;
    currentLetterIdx: number;
    currentWordIdx: number;
    currentVerseIdx: number;
    isPlaying: boolean;
}
