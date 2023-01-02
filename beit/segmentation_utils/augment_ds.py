import random

import os
import torchvision.transforms as T
from torchvision.io import read_image
import torch
import glob

IMG_DIR = '/data/pwojcik/images_he/'

if __name__ == '__main__':
    files = glob.glob(os.path.join(IMG_DIR, '*'))
    files = [f for f in files if not 'bis' in f and not 'aug' in f]
    c, r = [], []
    for f in files:
        n = f.split('-')
        c.append(int(n[-1].strip('.png')[1:]))
        r.append(int(n[-2][1:]))
    mc = max(c)
    mr = max(r)
    for ic in range(mc + 1):
        for jr in range(mr + 1):
            lu = ic, jr
            ru = ic + 1, jr
            ld = ic, jr + 1
            rd = ic + 1, jr + 1
            flu = f"wsi_001-tile-r{lu[0]}-c{lu[1]}.png"
            fru = f"wsi_001-tile-r{ru[0]}-c{ru[1]}.png"
            fld = f"wsi_001-tile-r{ld[0]}-c{ld[1]}.png"
            frd = f"wsi_001-tile-r{rd[0]}-c{rd[1]}.png"
            files = [flu, fru, fld, frd]
            all = True
            for f in files:
                if not os.path.exists(os.path.join(IMG_DIR, f)):
                    all = False
                    break
            if not all:
                continue
            print(ic, jr)
            images = [read_image(os.path.join(IMG_DIR, f)) for f in files]
            image1 = torch.cat([images[0], images[1]], dim=1)
            image2 = torch.cat([images[2], images[3]], dim=1)
            image = torch.cat([image1, image2], dim=2)
            #image_boxes = T.ToPILImage()(image)
            #canvas = ImageDraw.Draw(image_boxes)
            for cc in range(10):
                a = random.randint(10, 512)
                b = random.randint(10, 512)
                crop = image[:, a:(a + 512), b:(b + 512)]
                shape = [a, b, a + 512, b + 512]
                #canvas.rectangle(shape, outline="blue")
                assert(crop.shape == (3, 512, 512))
                crop = T.ToPILImage()(crop)
                crop.save(os.path.join(IMG_DIR, f"wsi_001-tile-r{lu[0]}-c{lu[0]}_{cc}aug.png"))
                #crop.show()
            #image_boxes.show()



