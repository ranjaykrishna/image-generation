from PIL import Image, ImageChops

import os

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__=='__main__':
    image_dir = 'downloads/madhubani/'
    trimmed_dir = 'downloads/trimmed/'
    if not os.path.exists(trimmed_dir):
        os.makedirs(trimmed_dir)
    for filename in os.listdir(image_dir):
        if '.png' in filename or '.jpg' in filename:
            try:
                im = Image.open(os.path.join(image_dir, filename))
                im = trim(im)
                im.save(os.path.join(trimmed_dir, filename), 'JPEG')
                print("Trimmed %s" % filename)
            except Exception as e:
                print(e)
