import numpy as np
from PIL import Image, ImageDraw, ImageFont


def display_image(img, targets):
    fs = 10
    draw = ImageDraw.Draw(img, mode="RGB")
    font = ImageFont.truetype("arial.ttf", fs)
    for target in targets:
        draw.rectangle([(target["box_x"], target["box_y"]), 
                        (target["box_x"]+target["box_width"], target["box_y"]+target["box_height"])],
                       outline="#00FF00", fill=None)
        draw.point((target["center_x"], target["center_y"]), fill="#FF0000")
        draw.text((target["center_x"], target["center_y"]), 
                    str(target["class"]), fill="#0000FF", font=font)
    img.show()
