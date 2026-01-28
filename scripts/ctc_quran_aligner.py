#!/usr/bin/env python3
"""
CTC Forced Aligner for Quran
Uses MahmoudAshraf97/ctc-forced-aligner for precise character-level alignment.
"""
import json
import torch
from pathlib import Path
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# Config
PROJECT_ROOT = Path(__file__).parent.parent
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_001.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # Lower for CPU

def load_quran_text(surah_num: int) -> str:
    """Load Quran text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)

def main():
    surah = 1
    print(f"=== CTC Forced Aligner for Surah {surah} ===")
    
    # 1. Load alignment model
    print("Loading alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    
    # 2. Load audio
    print("Loading audio...")
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    
    # 3. Get Quran text
    text = load_quran_text(surah)
    print(f"Text length: {len(text)} chars")
    
    # 4. Generate emissions (acoustic features from audio)
    print("Generating emissions...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    
    # 5. Preprocess text for alignment
    # romanize=False for Arabic since we want native script matching
    # language="ara" for Arabic
    print("Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,  # CTC aligner uses romanization for alignment
        language="ara",
    )
    
    # 6. Get alignments
    print("Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )
    
    # 7. Get spans
    spans = get_spans(tokens_starred, segments, blank_token)
    
    # 8. Post-process results
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"Got {len(word_timestamps)} word alignments")
    
    # 9. Convert to character-level timing
    # Each word_timestamp has 'text', 'start', 'end', 'score'
    char_timings = []
    for wt in word_timestamps:
        word = wt['text']
        start = wt['start']
        end = wt['end']
        duration = end - start
        char_dur = duration / len(word) if word else 0
        
        for i, char in enumerate(word):
            if not char.isspace():
                char_timings.append({
                    "idx": len(char_timings),
                    "char": char,
                    "start": start + i * char_dur,
                    "end": start + (i + 1) * char_dur,
                })
    
    print(f"Total chars: {len(char_timings)}")
    
    # 10. Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"letter_timing_{surah}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(char_timings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Print first 10 for verification
    print("\nFirst 10 characters:")
    for ct in char_timings[:10]:
        print(f"  {ct['idx']}: '{ct['char']}' @ {ct['start']:.3f}s - {ct['end']:.3f}s")

if __name__ == "__main__":
    main()
