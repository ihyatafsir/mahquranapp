import { useState, useEffect, useRef, useCallback } from 'react';
import type { LetterTiming, SyncState } from '../types/quran';

// Binary search to find current letter index based on audio time
// Returns -1 if playback is in a long breath pause / silence gap
export function findCurrentLetterIdx(timing: LetterTiming[], currentTime: number): number {
    if (!timing || timing.length === 0) return -1;
    if (currentTime < timing[0].start) return -1;

    let left = 0;
    let right = timing.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const letter = timing[mid];

        if (currentTime >= letter.start && currentTime < letter.end) {
            return mid;
        }

        if (currentTime < letter.start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // If currentTime falls slightly past the letter end but before a new letter starts
    if (right >= 0 && right < timing.length) {
        const prev = timing[right];
        const next = right + 1 < timing.length ? timing[right + 1] : null;
        
        // If next letter is within 0.15s, keep prev active for smooth visual legato
        if (currentTime >= prev.start && (next ? currentTime < next.start : currentTime < prev.end + 0.3)) {
            return right;
        }
    }

    return -1;
}

export function useLetterSync(
    audioRef: React.RefObject<HTMLAudioElement | null>,
    letterTiming: LetterTiming[]
) {
    const [syncState, setSyncState] = useState<SyncState>({
        currentTime: 0,
        duration: 0,
        currentLetterIdx: -1,
        currentWordIdx: -1,
        currentVerseIdx: 0,
        isPlaying: false,
    });

    const animationFrameRef = useRef<number | null>(null);
    const lastLetterIdxRef = useRef<number>(-1);
    const lastWordIdxRef = useRef<number>(-1);
    const lastVerseIdxRef = useRef<number>(-1);

    const updateSync = useCallback(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const currentTime = audio.currentTime;
        const duration = audio.duration || 0;

        if (letterTiming.length === 0) {
            setSyncState(prev => ({
                ...prev,
                currentTime,
                duration,
                isPlaying: !audio.paused,
            }));
            return;
        }

        const letterIdx = findCurrentLetterIdx(letterTiming, currentTime);
        let wordIdx = -1;
        let verseIdx = 0;

        if (letterIdx >= 0 && letterIdx < letterTiming.length) {
            const letter = letterTiming[letterIdx];
            wordIdx = letter.wordIdx ?? -1;
            verseIdx = typeof letter.verseIdx !== 'undefined'
                ? letter.verseIdx
                : (letter.ayah ? letter.ayah - 1 : 0);
        } else if (lastWordIdxRef.current >= 0) {
            // In a pause, keep current verse
            verseIdx = lastVerseIdxRef.current >= 0 ? lastVerseIdxRef.current : 0;
        }

        // Only trigger React state updates when indices change or audio time moves
        if (
            letterIdx !== lastLetterIdxRef.current ||
            wordIdx !== lastWordIdxRef.current ||
            verseIdx !== lastVerseIdxRef.current
        ) {
            lastLetterIdxRef.current = letterIdx;
            lastWordIdxRef.current = wordIdx;
            lastVerseIdxRef.current = verseIdx;

            setSyncState({
                currentTime,
                duration,
                currentLetterIdx: letterIdx,
                currentWordIdx: wordIdx,
                currentVerseIdx: verseIdx,
                isPlaying: !audio.paused,
            });
        }
    }, [audioRef, letterTiming]);

    useEffect(() => {
        const loop = () => {
            updateSync();
            animationFrameRef.current = requestAnimationFrame(loop);
        };

        animationFrameRef.current = requestAnimationFrame(loop);

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [updateSync]);

    return syncState;
}
