from PIL import Image, ImageDraw, ImageFont

image = Image.new("RGB", (640, 280), "white")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((80, 100), "PUMP P-101  PRESSURE 12.5 bar", fill="black", font=font)
draw.text((80, 220), "FLOW FT-204  125 m3/h", fill="black", font=font)
image.save("/tmp/pidocr_smoke.png")
