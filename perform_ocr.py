from calamari_ocr.ocr.predictor import Predictor

import cv2 as cv

import textAlignPreprocessing as preproc

fname = 'salzinnes_378'
raw_image = cv.imread('./png/{}_text.png'.format(fname))

img_bin, img_eroded, angle = preproc.preprocess_images(raw_image)
line_strips, lines_peak_locs, proj = preproc.identify_text_lines(img_eroded)

predictor = Predictor(checkpoint='./models/mcgill_salzinnes/1.ckpt')

# x, y, width, height
strips = []
for ls in line_strips:
    x, y, w, h = ls
    # WHY IS Y FIRST? WHAT'S THE DEAL WITH THIS
    strip = img_eroded[y:y + h, x:x + w]
    strips.append(strip)

results = []

for r in predictor.predict_raw(strips):
    results.append(r)
