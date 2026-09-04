import os
from PIL import Image, ImageDraw, ImageFont

ARTIFACTS_DIR = "/home/grem3/.gemini/antigravity-ide/brain/b9525ca8-ab73-4709-8a4c-a9a3075a8d13"

# Mobile Screen: 390px x 844px (iPhone 14/15 size)
img = Image.new("RGB", (390, 800), color=(11, 17, 33))
draw = ImageDraw.Draw(img)

font_arabic = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf", 22)
font_badge = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 9)
font_latin_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
font_latin = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)

# 1. Mobile Status Bar & Header
draw.text((15, 15), "9:41", font=font_latin_bold, fill=(255, 255, 255))
draw.text((15, 40), "1. AL-FATIHA (الفاتحة)", font=font_latin_bold, fill=(0, 255, 170))
draw.text((15, 58), "Sheikh Abdul Basit (Murattal)", font=font_latin, fill=(148, 163, 184))

# 2. Mobile Tajweed Live Inspector HUD Bar
draw.rounded_rectangle([15, 80, 375, 122], radius=10, fill=(15, 23, 42), outline=(0, 255, 170), width=1)
draw.ellipse([25, 96, 33, 104], fill=(0, 255, 170))
draw.text((42, 88), "ٱللَّهِ", font=font_arabic, fill=(0, 255, 170))
draw.rounded_rectangle([95, 93, 160, 110], radius=4, fill=(0, 0, 0, 100), outline=(0, 255, 170), width=1)
draw.text((99, 95), "غنة مشددة 2ح", font=font_badge, fill=(0, 255, 170))
draw.text((250, 95), "غنة الحرف المشدد", font=font_latin, fill=(148, 163, 184))

# 3. Mobile Waveform Sync Canvas
draw.rounded_rectangle([15, 130, 375, 185], radius=10, fill=(15, 23, 42), outline=(40, 55, 75), width=1)
# Draw mini wave
for i in range(70):
    x = 22 + i * 5
    import math
    h = (math.sin(i * 0.25) * 0.5 + 0.5) * 26 + 4
    is_past = (x <= 160)
    b_col = (0, 255, 170) if is_past else (51, 65, 85)
    draw.rectangle([x, 158 - h/2, x + 3, 158 + h/2], fill=b_col)

# Laser Playhead
draw.line([160, 130, 160, 185], fill=(255, 255, 255), width=2)
draw.ellipse([157, 155, 163, 161], fill=(0, 240, 255))

# 4. Mobile Quran Content Card with Flowing Word Pills & Visible Tajweed Colors
draw.rounded_rectangle([15, 195, 375, 760], radius=12, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
draw.text((25, 208), "Ayah 1:1", font=font_latin_bold, fill=(0, 255, 170))

# Ayah 1 Word Pills Row
words = [
    {"ar": "بِسْمِ", "col": (0, 255, 170), "w": 68, "active": False, "past": True, "fill": 100},
    {"ar": "ٱللَّهِ", "col": (0, 255, 170), "w": 70, "active": True, "past": False, "fill": 70, "badge": "غنة 2ح"},
    {"ar": "ٱلرَّحْمَٰنِ", "col": (103, 232, 249), "w": 95, "active": False, "past": False, "fill": 0}, # Madd Tabiee
    {"ar": "ٱلرَّحِيمِ", "col": (56, 189, 248), "w": 85, "active": False, "past": False, "fill": 0}, # Madd Arid
]

wx = 360
wy = 235
for w in words:
    ww = w["w"]
    rx1 = wx - ww
    rx2 = wx
    if w["active"]:
        draw.rounded_rectangle([rx1, wy, rx2, wy + 58], radius=8, fill=(0, 255, 170, 30), outline=(0, 255, 170), width=2)
        if "badge" in w:
            draw.text((rx1 + 14, wy - 10), w["badge"], font=font_badge, fill=(0, 255, 170))
    elif w["past"]:
        draw.rounded_rectangle([rx1, wy, rx2, wy + 58], radius=8, fill=(18, 28, 48), outline=(0, 255, 170, 70), width=1)
    else:
        draw.rounded_rectangle([rx1, wy, rx2, wy + 58], radius=8, fill=(20, 28, 45), outline=(40, 55, 75), width=1)

    # Arabic Text
    draw.text((rx1 + 10, wy + 8), w["ar"], font=font_arabic, fill=w["col"])

    # Micro waveform
    for b in range(8):
        bx = rx1 + 8 + b * (ww - 16) / 8
        is_f = (b / 8) * 100 <= w["fill"]
        draw.rectangle([bx, wy + 48, bx + 4, wy + 52], fill=(0, 255, 170) if is_f else (45, 55, 75))

    wx -= (ww + 6)

# Ayah 7 Madd Lazim Word Pill Preview
draw.text((25, 315), "Ayah 1:7 (Madd Lazim 6H Preview)", font=font_latin_bold, fill=(0, 240, 255))
draw.rounded_rectangle([25, 340, 365, 410], radius=10, fill=(0, 240, 255, 25), outline=(0, 240, 255), width=2)
draw.text((210, 348), "وَلَا ٱلضَّآلِّينَ", font=font_arabic, fill=(0, 240, 255))
draw.rounded_rectangle([40, 355, 150, 385], radius=6, fill=(15, 23, 42), outline=(0, 240, 255), width=1)
draw.text((48, 362), "مد لازم كلمي 6ح ⏱", font=font_badge, fill=(0, 240, 255))

# Mobile Tajweed Guide Legend
draw.rounded_rectangle([25, 430, 365, 530], radius=10, fill=(15, 23, 42), outline=(40, 55, 75), width=1)
draw.text((35, 442), "TAJWEED MUSHAF LEGEND", font=font_latin_bold, fill=(0, 255, 170))
legend = [
    ("مد لازم (6H)", (0, 240, 255)),
    ("غنة مشددة (2H)", (0, 255, 170)),
    ("مد طبيعي (2H)", (103, 232, 249)),
    ("قلقلة", (251, 191, 36)),
    ("تفخيم", (192, 132, 252)),
]
ly = 465
for l_name, l_col in legend:
    draw.ellipse([35, ly + 2, 43, ly + 10], fill=l_col)
    draw.text((50, ly), l_name, font=font_latin, fill=(203, 213, 225))
    ly += 12

out_path = os.path.join(ARTIFACTS_DIR, "mobile_ui_visual_inspection.png")
img.save(out_path)
print(f"Generated Mobile UI visual inspection frame: {out_path}")
