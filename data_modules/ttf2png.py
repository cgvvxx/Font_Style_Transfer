from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import os


# Convert ttf to png
# font_path is the path where ttf files exist
def ttf2png():
    fonts = os.listdir("./ttfs")
    if 'KumarOne-Regular.ttf' in fonts:  # remove unrecognible fonts
        fonts.remove('KumarOne-Regular.ttf')
    if 'KumarOneOutline-Regular.ttf' in fonts:
        fonts.remove('KumarOneOutline-Regular.ttf')

    # capital letters
    co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
    co = co.split(" ")
    start = "0041"
    end = "005A"
    Latin_Capital_letters = ["00" + a + b
                             for a in co
                             for b in co]
    Capitals = np.array(Latin_Capital_letters)

    s = np.where(start == Capitals)[0][0]
    e = np.where(end == Capitals)[0][0]

    Capitals = Capitals[s:e + 1]
    Capitals = list(Capitals)

    # small letters
    co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
    co = co.split(" ")
    start = "0061"
    end = "007A"
    Latin_Small_letters = ["00" + a + b
                           for a in co
                           for b in co]

    Smalls = np.array(Latin_Small_letters)

    s = np.where(start == Smalls)[0][0]
    e = np.where(end == Smalls)[0][0]

    Smalls = Smalls[s:e + 1]
    Smalls = list(Smalls)

    # numbers
    co = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
    co = co.split(" ")
    start = "0030"
    end = "0039"

    Numbers = ["00" + a + b
               for a in co
               for b in co]

    Numbers = np.array(Numbers)

    s = np.where(start == Numbers)[0][0]
    e = np.where(end == Numbers)[0][0]

    Numbers = Numbers[s:e + 1]
    Numbers = list(Numbers)

    Characters = Capitals + Smalls + Numbers

    for ttf in fonts:
        path = "./ttf2png/" + ttf + "/"

        os.makedirs(path, exist_ok = True)

        for uni in tqdm(Characters):
            unicodeChars = chr(int(uni, 16))
            font = ImageFont.truetype(font="./ttfs/" + ttf, size=100)
            x, y = font.getsize(unicodeChars)
            theImage = Image.new('RGB', (x+3, y+3), color='white')
            theDrawPad = ImageDraw.Draw(theImage)
            theDrawPad.text((0,0), unicodeChars[0], font=font, fill='black')
            img_path = path + "/" + ttf[:-4] + "_" + uni
            theImage.save('{}.png'.format(img_path))
