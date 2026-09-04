import json
import os
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "/home/grem3/mahquranapp/public/data"
ARTIFACTS_DIR = "/home/grem3/.gemini/antigravity-ide/brain/b9525ca8-ab73-4709-8a4c-a9a3075a8d13"

with open(os.path.join(DATA_DIR, "waveforms", "abdul_basit_murattal_surah_1.json")) as f:
    wf = json.load(f)
peaks = wf["peaks"]

with open(os.path.join(DATA_DIR, "abdul_basit_murattal", "letter_timing_1.json")) as f:
    timing = json.load(f)

img = Image.new("RGB", (1000, 380), color=(11, 17, 33))
draw = ImageDraw.Draw(img)

font_arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", 20)
font_latin = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
font_latin_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

draw.text((30, 20), "WAVEPLACE VOCALISATION ALIGNER • Waveform-Centric Letter Placement", font=font_latin, fill=(0, 255, 170))
draw.text((30, 42), "Every known canonical Quranic letter is physically locked to its acoustic wave peak", font=font_latin_sm, fill=(148, 163, 184))

# Draw Waveform Canvas Box
draw.rounded_rectangle([30, 70, 970, 350], radius=16, fill=(15, 23, 42), outline=(0, 255, 170, 60), width=1)

# Focus on Ayah 1 (0.0s to 4.2s)
ayah1_peaks = peaks[:420] # 4.2s
width = 940 - 60
height = 240
centerY = 70 + height / 2 + 20

# Draw Waveform Bars
for i, amp in enumerate(ayah1_peaks):
    x = 60 + (i / len(ayah1_peaks)) * width
    bar_h = max(3, amp * 130)
    is_past = (x <= 60 + (2.0 / 4.2) * width) # t = 2.0s
    color = (0, 255, 170) if is_past else (51, 65, 85)
    draw.rectangle([x, centerY - bar_h/2, x + 2, centerY + bar_h/2], fill=color)

# Draw Letter Tags above wave crests
ayah1_letters = [l for l in timing if l["ayah"] == 1]
for l in ayah1_letters:
    p_time = l.get("peakTime", (l["start"] + l["end"])/2)
    x = 60 + (p_time / 4.2) * width
    ch = l["char"]
    
    is_active = (2.0 >= l["start"] and 2.0 < l["end"])
    is_past = (2.0 >= l["end"])
    
    # Drop line to wave peak
    line_col = (0, 240, 255) if is_active else ((0, 255, 170, 80) if is_past else (70, 85, 105))
    draw.line([x, 100, x, centerY - 20], fill=line_col, width=1 if not is_active else 2)
    
    # Letter Tag
    tag_col = (0, 255, 170) if is_past or is_active else (100, 116, 139)
    draw.text((x - 8, 85), ch, font=font_arabic, fill=tag_col)

# Laser Playhead at t = 2.0s
playhead_x = 60 + (2.0 / 4.2) * width
draw.line([playhead_x, 70, playhead_x, 350], fill=(255, 255, 255), width=2)
draw.ellipse([playhead_x - 5, centerY - 5, playhead_x + 5, centerY + 5], fill=(0, 240, 255))

out_path = os.path.join(ARTIFACTS_DIR, "waveplace_visual_inspection.png")
img.save(out_path)
print(f"Generated Waveplace visual inspection frame: {out_path}")
