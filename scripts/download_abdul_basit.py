#!/usr/bin/env python3
"""
Download Abdul Basit Mujawwad audio files from quranicaudio.com
For use with batch_lisan_aligner.py wave detection
"""
import urllib.request
from pathlib import Path
import sys

OUTPUT_DIR = Path(__file__).parent.parent / "public/audio/abdul_basit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_surah(surah_num: int) -> bool:
    """Download a single surah from quranicaudio.com"""
    output_path = OUTPUT_DIR / f"surah_{surah_num:03d}.mp3"
    
    if output_path.exists() and output_path.stat().st_size > 10000:
        print(f"  [SKIP] Surah {surah_num} already exists ({output_path.stat().st_size//1024}KB)")
        return True
    
    # quranicaudio.com CDN URL for Abdul Basit Mujawwad
    url = f"https://download.quranicaudio.com/qdc/abdul_baset/mujawwad/{surah_num}.mp3"
    print(f"  [DOWNLOAD] Surah {surah_num} from {url}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        file_size = output_path.stat().st_size
        if file_size < 10000:  # Less than 10KB = error
            print(f"  ✗ File too small ({file_size} bytes)")
            output_path.unlink()
            return False
        print(f"  ✓ Downloaded {output_path.name} ({file_size//(1024*1024)}MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 114
    
    print("=" * 60)
    print("Abdul Basit Mujawwad Audio Downloader")
    print(f"Surahs {start} - {end}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for surah in range(start, end + 1):
        if download_surah(surah):
            success += 1
        else:
            failed += 1
    
    print(f"\n=== Complete ===")
    print(f"Downloaded: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
