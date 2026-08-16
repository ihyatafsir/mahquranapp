import { useState, useEffect, useRef, useCallback } from 'react';
import type { LetterTiming, SyncState } from '../types/quran';

// High-precision binary search: returns the exact active letter index
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
        } else if (currentTime < letter.start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // If currentTime falls in a gap between letters, return previous letter
    if (right >= 0 && right < timing.length && currentTime >= timing[right].start) {
        return right;
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

    const lastLetterIdxRef = useRef<number>(-1);
    const lastWordIdxRef = useRef<number>(-1);
    const lastVerseIdxRef = useRef<number>(0);
    const animationFrameRef = useRef<number | null>(null);

    const checkSync = useCallback(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const currentTime = audio.currentTime;
        const duration = audio.duration || 0;

        if (letterTiming.length > 0) {
            const letterIdx = findCurrentLetterIdx(letterTiming, currentTime);
            const letter = letterIdx >= 0 ? letterTiming[letterIdx] : null;
            const wordIdx = letter ? (letter.wordIdx ?? -1) : -1;
            const verseIdx = letter ? (letter.verseIdx ?? 0) : 0;

            if (
                letterIdx !== lastLetterIdxRef.current ||
                wordIdx !== lastWordIdxRef.current ||
                verseIdx !== lastVerseIdxRef.current
            ) {
                lastLetterIdxRef.current = letterIdx;
                lastWordIdxRef.current = wordIdx;
                lastVerseIdxRef.current = verseIdx;

                setSyncState(prev => ({
                    ...prev,
                    currentTime,
                    duration,
                    currentLetterIdx: letterIdx,
                    currentWordIdx: wordIdx,
                    currentVerseIdx: verseIdx,
                    isPlaying: !audio.paused,
                }));
            }
        }

        if (!audio.paused) {
            animationFrameRef.current = requestAnimationFrame(checkSync);
        }
    }, [audioRef, letterTiming]);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handlePlay = () => {
            setSyncState(prev => ({ ...prev, isPlaying: true }));
            animationFrameRef.current = requestAnimationFrame(checkSync);
        };

        const handlePause = () => {
            setSyncState(prev => ({ ...prev, isPlaying: false }));
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };

        const handleSeeked = () => {
            lastLetterIdxRef.current = -2; // Force re-eval
            checkSync();
        };

        const handleLoadedMetadata = () => {
            setSyncState(prev => ({
                ...prev,
                duration: audio.duration || 0,
            }));
        };

        audio.addEventListener('play', handlePlay);
        audio.addEventListener('pause', handlePause);
        audio.addEventListener('seeked', handleSeeked);
        audio.addEventListener('loadedmetadata', handleLoadedMetadata);

        return () => {
            audio.removeEventListener('play', handlePlay);
            audio.removeEventListener('pause', handlePause);
            audio.removeEventListener('seeked', handleSeeked);
            audio.removeEventListener('loadedmetadata', handleLoadedMetadata);

            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [audioRef, checkSync]);

    return syncState;
}
