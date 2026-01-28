import json
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_surah_text(verses_data, surah_num):
    surah_data = verses_data[str(surah_num)]
    full_text = ""
    for ayah in surah_data:
        full_text += ayah['text']
    return full_text

def analyze_mismatch():
    print("Loading data...")
    try:
        current_timing = load_json('public/data/letter_timing_1.json')
        backup_timing = load_json('public/data/letter_timing_1.backup.json')
        verses = load_json('public/data/verses_v4.json')
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return

    quran_text = get_surah_text(verses, 1)
    
    # Filter backup timing to remove Isti'aatha (first 4 words in backup are Isti'aatha usually)
    # Backup WordIdx 0,1,2,3,4? Let's check text.
    # Backup starts with "أ" at 0.031. Current starts at 3.879 (WordIdx 5 in backup).
    # So Backup[0..?] is extra.
    
    print(f"\n--- Data Summary ---")
    print(f"Quran Text Length (chars): {len(quran_text)}")
    print(f"Current Timing Entries: {len(current_timing)}")
    print(f"Backup Timing Entries: {len(backup_timing)}")
    
    print(f"\n--- Detailed Comparison (First 100 items) ---")
    print(f"{'Idx':<5} | {'Quran':<5} | {'Current':<10} | {'Backup':<10} | {'Match?'}")
    print("-" * 50)
    
    # We compare Quran Text vs Current Timing mostly
    limit = min(len(quran_text), len(current_timing), 100)
    
    for i in range(limit):
        q_char = quran_text[i]
        c_item = current_timing[i]
        c_char = c_item['char']
        
        # Try to find corresponding item in backup if possible
        # Backup has offset. "Bismi" starts at index where wordIdx=5?
        # Let's just print raw backup at same index for now, usually it will be shifted.
        b_char = backup_timing[i]['char'] if i < len(backup_timing) else "N/A"
        
        match = "YES" if q_char == c_char else "NO"
        
        print(f"{i:<5} | {q_char:<5} | {c_char:<10} | {b_char:<10} | {match}")

    print("\n--- Start of Files ---")
    print(f"Quran Start: {quran_text[:20]}")
    print(f"Current Timing Start: {[x['char'] for x in current_timing[:5]]}")
    print(f"Backup Timing Start: {[x['char'] for x in backup_timing[:5]]}")

    # Check alignment mismatch point
    mismatch_idx = -1
    for i in range(min(len(quran_text), len(current_timing))):
        if quran_text[i] != current_timing[i]['char']:
            mismatch_idx = i
            break
            
    if mismatch_idx != -1:
        print(f"\n[!] First mismatch at index {mismatch_idx}")
        print(f"Quran: {quran_text[mismatch_idx]} (Unicode: {ord(quran_text[mismatch_idx])})")
        print(f"Timing: {current_timing[mismatch_idx]['char']} (Unicode: {ord(current_timing[mismatch_idx]['char'])})")
        
        # Show context
        start = max(0, mismatch_idx - 5)
        end = min(len(quran_text), mismatch_idx + 5)
        print(f"Context Quran: {quran_text[start:end]}")
        print(f"Context Timing: {[x['char'] for x in current_timing[start:end]]}")

if __name__ == "__main__":
    analyze_mismatch()
