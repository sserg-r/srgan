import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
        
def text_phantom(text, size):
   
    font = 'Gargi'
    
    pil_font = ImageFont.truetype(font + ".ttf", size=size[1]*2 // len(text), encoding="unic")
    text_width, text_height = pil_font.getsize(text)
   
    canvas = Image.new('RGB', [size[1],size[0]], (255, 255, 255))
    
    draw = ImageDraw.Draw(canvas)
    offset = ((size[1] - text_width) // 2,
              (size[0] - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)
   
    return (255 - np.asarray(canvas)) / 255.0