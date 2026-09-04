import { useState, useEffect, useRef, useCallback } from 'react';
import type { LetterTiming, SyncState } from '../types/quran';

/**
 * High-precision temporal alignment search with O(1) locality caching & binary search fallback.
 * Handles continuous playback, micro-legato letter smoothing, and abrupt seeking.
 */
export function findCurrentLetterIdx(
    timing: LetterTiming[],
    currentTime: number,
    offsetMs: number = 0,
    hintIdx: number = -1
): number {
    if (!timing || timing.length === 0) return -1;
    const adjustedTime = Math.max(0, currentTime + (offsetMs / 1000));
    if (adjustedTime < timing[0].start - 0.05) return -1;

    // Fast O(1) temporal locality check for continuous forward playback
    if (hintIdx >= 0 && hintIdx < timing.length) {
        const currentLetter = timing[hintIdx];
        // 1. Still inside current letter
        if (adjustedTime >= currentLetter.start && adjustedTime < currentLetter.end) {
            return hintIdx;
        }

        // 2. Advanced to immediate next letter
        if (hintIdx + 1 < timing.length) {
            const nextLetter = timing[hintIdx + 1];
            if (adjustedTime >= nextLetter.start && adjustedTime < nextLetter.end) {
                return hintIdx + 1;
            }
            // 3. In micro-gap between current and next letter (< 0.25s)
            if (adjustedTime >= currentLetter.end && adjustedTime < nextLetter.start && (nextLetter.start - currentLetter.end) <= 0.25) {
                return hintIdx;
            }
        } else if (adjustedTime >= currentLetter.end && adjustedTime < currentLetter.end + 0.3) {
            return hintIdx;
        }
    }

    // Binary search fallback for seeking / rewinding / initial alignment
    let left = 0;
    let right = timing.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const letter = timing[mid];

        if (adjustedTime >= letter.start && adjustedTime < letter.end) {
            return mid;
        }

        if (adjustedTime < letter.start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // Micro-legato smoothing fallback
    if (right >= 0 && right < timing.length) {
        const prev = timing[right];
        const next = right + 1 < timing.length ? timing[right + 1] : null;

        if (next && adjustedTime < next.start && (next.start - prev.end) <= 0.25) {
            return right;
        } else if (!next && adjustedTime < prev.end + 0.3) {
            return right;
        }
    }

    return -1;
}

export function useLetterSync(
    audioRef: React.RefObject<HTMLAudioElement | null>,
    letterTiming: LetterTiming[],
    offsetMs: number = 0,
    stopAtTimeRef?: React.MutableRefObject<number | null>,
    onStopReached?: () => void
) {
    const [syncState, setSyncState] = useState<SyncState>({
        currentTime: 0,
        duration: 0,
        currentLetterIdx: -1,
        currentWordIdx: -1,
        currentVerseIdx: 0,
        letterProgress: 0,
        isPlaying: false,
    });

    const animationFrameRef = useRef<number | null>(null);
    const lastLetterIdxRef = useRef<number>(-1);
    const lastWordIdxRef = useRef<number>(-1);
    const lastVerseIdxRef = useRef<number>(-1);
    const lastTimeRef = useRef<number>(0);
    const lastPlayStateRef = useRef<boolean>(false);

    const updateSync = useCallback(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const currentTime = audio.currentTime;
        const duration = audio.duration || 0;
        const isPlaying = !audio.paused;

        // Auto-pause boundary check for single-word auditioning (WhisperX precision)
        if (stopAtTimeRef && stopAtTimeRef.current !== null && isPlaying) {
            if (currentTime >= stopAtTimeRef.current) {
                audio.pause();
                stopAtTimeRef.current = null;
                if (onStopReached) {
                    onStopReached();
                }
            }
        }

        if (letterTiming.length === 0) {
            if (Math.abs(currentTime - lastTimeRef.current) > 0.05 || isPlaying !== lastPlayStateRef.current) {
                lastTimeRef.current = currentTime;
                lastPlayStateRef.current = isPlaying;
                setSyncState(prev => ({
                    ...prev,
                    currentTime,
                    duration,
                    letterProgress: 0,
                    isPlaying,
                }));
            }
            return;
        }

        const adjustedTime = Math.max(0, currentTime + (offsetMs / 1000));
        const letterIdx = findCurrentLetterIdx(letterTiming, currentTime, offsetMs, lastLetterIdxRef.current);
        let wordIdx = -1;
        let verseIdx = 0;
        let letterProgress = 0;

        if (letterIdx >= 0 && letterIdx < letterTiming.length) {
            const letter = letterTiming[letterIdx];
            wordIdx = letter.wordIdx ?? -1;
            verseIdx = typeof letter.verseIdx !== 'undefined'
                ? letter.verseIdx
                : (letter.ayah ? letter.ayah - 1 : 0);

            const letterDuration = Math.max(0.005, letter.end - letter.start);
            letterProgress = Math.max(0, Math.min(1, (adjustedTime - letter.start) / letterDuration));
        } else if (lastWordIdxRef.current >= 0) {
            wordIdx = lastWordIdxRef.current;
            verseIdx = lastVerseIdxRef.current >= 0 ? lastVerseIdxRef.current : 0;
        }

        const indicesChanged = (
            letterIdx !== lastLetterIdxRef.current ||
            wordIdx !== lastWordIdxRef.current ||
            verseIdx !== lastVerseIdxRef.current
        );

        const timeMoved = Math.abs(currentTime - lastTimeRef.current) > 0.03;
        const playStateChanged = isPlaying !== lastPlayStateRef.current;

        if (indicesChanged || (isPlaying && timeMoved) || playStateChanged) {
            lastLetterIdxRef.current = letterIdx;
            lastWordIdxRef.current = wordIdx;
            lastVerseIdxRef.current = verseIdx;
            lastTimeRef.current = currentTime;
            lastPlayStateRef.current = isPlaying;

            setSyncState({
                currentTime,
                duration,
                currentLetterIdx: letterIdx,
                currentWordIdx: wordIdx,
                currentVerseIdx: verseIdx,
                letterProgress,
                isPlaying,
            });
        }
    }, [audioRef, letterTiming, offsetMs, stopAtTimeRef, onStopReached]);

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
