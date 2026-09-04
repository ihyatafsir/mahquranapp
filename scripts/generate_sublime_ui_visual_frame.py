import json
import os
from PIL import Image, ImageDraw, ImageFont

ARTIFACTS_DIR = "/home/grem3/.gemini/antigravity-ide/brain/b9525ca8-ab73-4709-8a4c-a9a3075a8d13"

img = Image.new("RGB", (1000, 460), color=(11, 17, 33))
draw = ImageDraw.Draw(img)

font_arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", 32)
font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
font_latin = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
font_latin_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

draw.text((30, 20), "SUBLIME QURANIC WORD-WAVEFORM PILL & TAJWEED UI", font=font_latin, fill=(0, 255, 170))
draw.text((30, 42), "Embedded Micro-Waveform Ribbons + Classical Tajweed Rule Floating Badges", font=font_latin_sm, fill=(148, 163, 184))

# Words of Ayah 1
words_data = [
    {"text": "بِسْمِ", "w": 180, "active": False, "past": True, "fill": 100, "badge": ""},
    {"text": "ٱللَّهِ", "w": 190, "active": True, "past": False, "fill": 65, "badge": "غنة 2ح • الشدة"},
    {"text": "ٱلرَّحْمَٰنِ", "w": 230, "active": False, "past": False, "fill": 0, "badge": ""},
    {"text": "ٱلرَّحِيمِ", "w": 210, "active": False, "past": False, "fill": 0, "badge": ""},
]

start_x = 940
y = 110

for w in words_data:
    width = w["w"]
    height = 115
    rect_x1 = start_x - width
    rect_x2 = start_x
    rect_y1 = y
    rect_y2 = y + height

    # Pill container
    if w["active"]:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(0, 255, 170, 25), outline=(0, 255, 170), width=2)
        # Floating badge
        if w["badge"]:
            draw.rounded_rectangle([rect_x1 + width//2 - 45, rect_y1 - 12, rect_x1 + width//2 + 45, rect_y1 + 10], radius=6, fill=(15, 23, 42), outline=(0, 255, 170), width=1)
            draw.text((rect_x1 + width//2 - 38, rect_y1 - 8), w["badge"], font=font_badge, fill=(0, 255, 170))
    elif w["past"]:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(18, 28, 48), outline=(0, 255, 170, 80), width=1)
    else:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(15, 23, 42), outline=(40, 55, 75), width=1)

    # Arabic Text
    text_color = (0, 255, 170) if w["past"] or w["active"] else (100, 116, 139)
    draw.text((rect_x1 + 25, rect_y1 + 15), w["text"], font=font_arabic, fill=text_color)

    # Micro-waveform ribbon at bottom of pill
    num_bars = 16
    bar_w = (width - 30) / num_bars
    for b in range(num_bars):
        bx = rect_x1 + 15 + b * bar_w
        import math
        bh = math.sin((b / (num_bars - 1)) * math.pi) * 14 + 3
        is_filled = (b / num_bars) * 100 <= w["fill"]
        b_col = (0, 255, 170) if is_filled else (40, 55, 75)
        draw.rectangle([bx, rect_y2 - 20 - bh, bx + bar_w - 2, rect_y2 - 20], fill=b_col)

    start_x -= (width + 18)

# Lower Tajweed Guide Banner
draw.rounded_rectangle([30, 270, 970, 420], radius=16, fill=(15, 23, 42), outline=(50, 65, 85), width=1)
draw.text((50, 290), "CLASSICAL TAJWEED COLOR-CODED VOCALISATION HELPER", font=font_latin, fill=(0, 255, 170))

rules = [
    ("مد لازم 6ح (Madd Lazim)", "#00f0ff"),
    ("غنة 2ح (Ghunnah)", "#00ffaa"),
    ("قلقلة (Qalqalah)", "#fbbf24"),
    ("تفخيم (Tafkheem)", "#c084fc"),
    ("همزة الوصل (Silent Wasl)", "#64748b"),
]

rx = 50
for r_name, r_col in rules:
    draw.ellipse([rx, 330, rx + 12, 342], fill=r_col)
    draw.text((rx + 18, 328), r_name, font=font_latin_sm, fill=(203, 213, 225))
    rx += 180

draw.text((50, 380), "Embedded micro-waveform tracks rhythm, amplitude, and breath duration in real-time.", font=font_latin_sm, fill=(148, 163, 184))

out_path = os.path.join(ARTIFACTS_DIR, "sublime_ui_visual_inspection.png")
img.save(out_path)
print(f"Generated Sublime UI visual inspection frame: {out_path}")
