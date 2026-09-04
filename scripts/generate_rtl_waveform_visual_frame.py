import os
from PIL import Image, ImageDraw, ImageFont

ARTIFACTS_DIR = "/home/grem3/.gemini/antigravity-ide/brain/b9525ca8-ab73-4709-8a4c-a9a3075a8d13"

img = Image.new("RGB", (1000, 360), color=(11, 17, 33))
draw = ImageDraw.Draw(img)

font_arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", 22)
font_latin = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
font_latin_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

draw.text((30, 20), "AUTHENTIC RIGHT-TO-LEFT (RTL) ACOUSTIC WAVEFORM CANVAS", font=font_latin, fill=(0, 255, 170))
draw.text((30, 42), "Time flows from Far Right (0.0s) to Far Left (End). Playhead sweeps Right-to-Left.", font=font_latin_sm, fill=(148, 163, 184))

# Waveform Box
draw.rounded_rectangle([30, 70, 970, 320], radius=14, fill=(15, 23, 42), outline=(0, 255, 170, 80), width=1)

# Labels
draw.text((900, 85), "[ 0.0s البداية ]", font=font_latin_sm, fill=(0, 255, 170))
draw.text((50, 85), "[ 41.8s النهاية ]", font=font_latin_sm, fill=(148, 163, 184))

# Draw RTL Waveform Bars
num_bars = 180
centerY = 205
playhead_x = 970 - int(0.35 * 900) # t = 35% through audio (Swept from right to left)

for i in range(num_bars):
    x = 940 - i * 5
    if x < 60: break
    import math
    h = abs(math.sin(i * 0.18) * math.cos(i * 0.08) * 55) + 8
    is_recited = (x >= playhead_x) # Recited is on the right!
    col = (0, 255, 170) if is_recited else (51, 65, 85)
    draw.rectangle([x, centerY - h/2, x + 3, centerY + h/2], fill=col)

# Laser Playhead
draw.line([playhead_x, 70, playhead_x, 320], fill=(255, 255, 255), width=2)
draw.ellipse([playhead_x - 5, centerY - 5, playhead_x + 5, centerY + 5], fill=(0, 240, 255))

# RTL Letter Tags over peaks
tags = [
    ("بِسْمِ", 910),
    ("ٱللَّهِ", 840),
    ("ٱلرَّحْمَٰنِ", 750),
    ("ٱلرَّحِيمِ", 660),
]

for tag_text, tag_x in tags:
    is_tag_recited = (tag_x >= playhead_x)
    col = (0, 255, 170) if is_tag_recited else (100, 116, 139)
    draw.line([tag_x, 115, tag_x, centerY - 20], fill=(0, 240, 255) if abs(tag_x - playhead_x) < 20 else (col[0], col[1], col[2], 80), width=1)
    draw.text((tag_x - 15, 95), tag_text, font=font_arabic, fill=col)

out_path = os.path.join(ARTIFACTS_DIR, "rtl_waveform_visual_inspection.png")
img.save(out_path)
print(f"Generated RTL waveform visual inspection frame: {out_path}")
