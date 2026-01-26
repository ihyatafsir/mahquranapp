import { useState, useEffect, useRef, useCallback } from 'react';
import type { LetterTiming, SyncState } from '../types/quran';

// Binary search to find current letter index based on audio time
function findCurrentLetterIdx(timing: LetterTiming[], currentTime: number): number {
    if (timing.length === 0) return -1;

    let left = 0;
    let right = timing.length - 1;
    let result = -1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        const letter = timing[mid];

        if (currentTime >= letter.start && currentTime <= letter.end) {
            return mid;
        } else if (currentTime < letter.start) {
            right = mid - 1;
        } else {
            result = mid;
            left = mid + 1;
        }
    }

    return result;
}

export function useLetterSync(
    audioRef: React.RefObject<HTMLAudioElement | null>,
    letterTiming: LetterTiming[]
) {
    const [syncState, setSyncState] = useState<SyncState>({
        currentTime: 0,
        currentLetterIdx: -1,
        currentWordIdx: -1,
        currentVerseIdx: 0,
        isPlaying: false,
    });

    const animationFrameRef = useRef<number | null>(null);

    const updateSync = useCallback(() => {
        const audio = audioRef.current;
        if (!audio || letterTiming.length === 0) return;

        const currentTime = audio.currentTime;
        const letterIdx = findCurrentLetterIdx(letterTiming, currentTime);

        const wordIdx = letterIdx >= 0 ? letterTiming[letterIdx]?.wordIdx ?? -1 : -1;

        setSyncState(prev => ({
            ...prev,
            currentTime,
            currentLetterIdx: letterIdx,
            currentWordIdx: wordIdx,
            isPlaying: !audio.paused,
        }));

        if (!audio.paused) {
            animationFrameRef.current = requestAnimationFrame(updateSync);
        }
    }, [audioRef, letterTiming]);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const handlePlay = () => {
            setSyncState(prev => ({ ...prev, isPlaying: true }));
            animationFrameRef.current = requestAnimationFrame(updateSync);
        };

        const handlePause = () => {
            setSyncState(prev => ({ ...prev, isPlaying: false }));
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };

        const handleTimeUpdate = () => {
            if (audio.paused) {
                updateSync();
            }
        };

        const handleSeeked = () => {
            updateSync();
        };

        audio.addEventListener('play', handlePlay);
        audio.addEventListener('pause', handlePause);
        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('seeked', handleSeeked);

        return () => {
            audio.removeEventListener('play', handlePlay);
            audio.removeEventListener('pause', handlePause);
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('seeked', handleSeeked);

            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [audioRef, updateSync]);

    return syncState;
}
