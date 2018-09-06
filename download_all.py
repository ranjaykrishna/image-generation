import json
import subprocess

image_folder = 'images'
keywords = json.load(open('keywords.json'))
for keyword in keywords:
    cmd = ['googleimagesdownload', '-k', keyword, '-l', '20000', '-t', 'photo',
           '-i', image_folder, '-f', 'jpg', '-nn', '--chromedriver',
           '/usr/lib/chromium-browser/chromedriver']
    subprocess.call(cmd)
