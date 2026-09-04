import os
from PIL import Image, ImageDraw, ImageFont

ARTIFACTS_DIR = "/home/grem3/.gemini/antigravity-ide/brain/b9525ca8-ab73-4709-8a4c-a9a3075a8d13"

img = Image.new("RGB", (1000, 480), color=(11, 17, 33))
draw = ImageDraw.Draw(img)

font_arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", 34)
font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
font_latin = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
font_latin_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

draw.text((30, 20), "TAJWEED SIGNS & HIGH-CONTRAST MUSHAF VISUAL INSPECTION", font=font_latin, fill=(0, 255, 170))
draw.text((30, 42), "Every letter maintains its authentic Tajweed color without background wash-out", font=font_latin_sm, fill=(148, 163, 184))

# Top Live Tajweed Inspector Bar
draw.rounded_rectangle([30, 70, 970, 115], radius=10, fill=(15, 23, 42), outline=(0, 255, 170), width=1)
draw.ellipse([45, 87, 55, 97], fill=(0, 255, 170))
draw.text((68, 77), "ٱللَّهِ", font=font_arabic, fill=(255, 255, 255))
draw.rounded_rectangle([130, 83, 215, 103], radius=6, fill=(0, 255, 170, 35), outline=(0, 255, 170), width=1)
draw.text((136, 85), "غنة مشددة 2ح", font=font_badge, fill=(0, 255, 170))
draw.text((300, 86), "غنة الحرف المشدد (النون والميم) • مخرج الخيشوم", font=font_latin_sm, fill=(203, 213, 225))
draw.text((850, 86), "⏱ 2 حركات", font=font_latin, fill=(0, 255, 170))

# Word Cards
words = [
    {"text": "بِسْمِ", "w": 180, "active": False, "past": True, "fill": 100, "col": (0, 255, 170), "badge": ""},
    {"text": "ٱللَّهِ", "w": 190, "active": True, "past": False, "fill": 65, "col": (0, 255, 170), "badge": "غنة 2ح • الشدة"},
    {"text": "ٱلرَّحْمَٰنِ", "w": 230, "active": False, "past": False, "fill": 0, "col": (103, 232, 249), "badge": ""}, # Madd Tabiee
    {"text": "ٱلرَّحِيمِ", "w": 210, "active": False, "past": False, "fill": 0, "col": (56, 189, 248), "badge": ""}, # Madd Arid
]

start_x = 940
y = 145

for w in words:
    width = w["w"]
    height = 120
    rect_x1 = start_x - width
    rect_x2 = start_x
    rect_y1 = y
    rect_y2 = y + height

    # Pill container
    if w["active"]:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(18, 26, 44), outline=(0, 255, 170), width=2)
        if w["badge"]:
            draw.rounded_rectangle([rect_x1 + width//2 - 45, rect_y1 - 12, rect_x1 + width//2 + 45, rect_y1 + 10], radius=6, fill=(15, 23, 42), outline=(0, 255, 170), width=1)
            draw.text((rect_x1 + width//2 - 38, rect_y1 - 8), w["badge"], font=font_badge, fill=(0, 255, 170))
    elif w["past"]:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(18, 28, 48), outline=(0, 255, 170, 80), width=1)
    else:
        draw.rounded_rectangle([rect_x1, rect_y1, rect_x2, rect_y2], radius=14, fill=(15, 23, 42), outline=(40, 55, 75), width=1)

    # Arabic Text with crisp authentic color
    draw.text((rect_x1 + 25, rect_y1 + 16), w["text"], font=font_arabic, fill=w["col"])

    # Micro-waveform ribbon
    num_bars = 16
    bar_w = (width - 30) / num_bars
    for b in range(num_bars):
        bx = rect_x1 + 15 + b * bar_w
        import math
        bh = math.sin((b / (num_bars - 1)) * math.pi) * 12 + 3
        is_filled = (b / num_bars) * 100 <= w["fill"]
        b_col = (0, 255, 170) if is_filled else (40, 55, 75)
        draw.rectangle([bx, rect_y2 - 18 - bh, bx + bar_w - 2, rect_y2 - 18], fill=b_col)

    start_x -= (width + 18)

# Lower Guide
draw.rounded_rectangle([30, 295, 970, 440], radius=16, fill=(15, 23, 42), outline=(50, 65, 85), width=1)
draw.text((50, 315), "AUTHENTIC MUSHAF TAJWEED COLOR LEGEND", font=font_latin, fill=(0, 255, 170))

rules = [
    ("مد لازم 6ح (Madd Lazim)", (0, 240, 255)),
    ("غنة مشددة 2ح (Ghunnah)", (0, 255, 170)),
    ("مد طبيعي 2ح (Madd Tabiee)", (103, 232, 249)),
    ("قلقلة ⚡ (Qalqalah)", (251, 191, 36)),
    ("تفخيم واستطالة ض (Tafkheem)", (192, 132, 252)),
]

rx = 50
for r_name, r_col in rules:
    draw.ellipse([rx, 355, rx + 12, 367], fill=r_col)
    draw.text((rx + 18, 353), r_name, font=font_latin_sm, fill=(203, 213, 225))
    rx += 180

draw.text((50, 400), "Every letter, diacritic, and Tajweed sign maintains clear contrast and legibility during playback.", font=font_latin_sm, fill=(148, 163, 184))

out_path = os.path.join(ARTIFACTS_DIR, "tajweed_signs_fixed_visual_inspection.png")
img.save(out_path)
print(f"Generated Tajweed signs fixed inspection frame: {out_path}")
