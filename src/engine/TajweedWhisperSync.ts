/**
 * TajweedWhisperSync - High-Precision Frame-Perfect Client-Side Sync Engine
 * Maps real-time audio playback timestamp (t) to exact Uthmani letter chunks
 * with zero temporal jitter and micro-interpolation.
 */

export interface TajweedLetterTiming {
  charIdx: number;
  charIdxInWord: number;
  char: string;
  start: number;
  end: number;
  duration: number;
  tajweedType?: string;
  wordIdx: number;
  ayah: number;
}

export class TajweedWhisperSync {
  private timingData: TajweedLetterTiming[] = [];
  public lastIdx: number = -1;

  constructor(timingData: TajweedLetterTiming[] = []) {
    this.timingData = timingData;
  }

  public setTimingData(data: TajweedLetterTiming[]) {
    this.timingData = data;
    this.lastIdx = -1;
  }

  /**
   * Fast Binary Search with Micro-Legato Lookahead (< 250ms)
   */
  public findActiveIndex(currentTime: number): number {
    const data = this.timingData;
    const len = data.length;
    if (len === 0) return -1;
    if (currentTime < data[0].start - 0.05) return -1;

    let left = 0;
    let right = len - 1;

    while (left <= right) {
      const mid = (left + right) >> 1;
      const item = data[mid];

      if (item.start <= currentTime && currentTime < item.end) {
        this.lastIdx = mid;
        return mid;
      }

      if (currentTime < item.start) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Micro-Legato bridging for natural breath pauses
    if (right >= 0 && right < len) {
      const prev = data[right];
      const next = right + 1 < len ? data[right + 1] : null;

      if (next && currentTime < next.start && (next.start - prev.end) <= 0.25) {
        return right;
      }
      if (!next && currentTime < prev.end + 0.3) {
        return right;
      }
    }

    return -1;
  }
}
