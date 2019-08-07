import PIL
import pickle
from PIL import Image, ImageDraw, ImageFont
import ElementTree as ET
import numpy as np
import textAlignPreprocessing as preproc
import gamera.core as gc
gc.init_gamera()


def intersect(bb1, bb2):
    '''
    takes two bounding boxes as an argument. if they overlap, return the area of the overlap;
    else, return False
    '''
    lr1 = bb1['lr']
    ul1 = bb1['ul']
    lr2 = bb2['lr']
    ul2 = bb2['ul']

    dx = min(lr1[0], lr2[0]) - max(ul1[0], ul2[0])
    dy = min(lr1[1], lr2[1]) - max(ul1[1], ul2[1])
    if (dx > 0) and (dy > 0):
        return dx*dy
    else:
        return False


def IOU(bb1, bb2):
    '''
    intersection over union between two bounding boxes
    '''
    lr1 = bb1['lr']
    ul1 = bb1['ul']
    lr2 = bb2['lr']
    ul2 = bb2['ul']

    # first find area of intersection:
    new_ulx = max(ul1[0], ul2[0])
    new_uly = max(ul1[1], ul2[1])
    new_lrx = min(lr1[0], lr2[0])
    new_lry = min(lr1[1], lr2[1])

    area_int = (new_lrx - new_ulx) * (new_lry - new_uly)
    area_1 = (lr1[0] - ul1[0]) * (lr1[1] - ul1[1])
    area_2 = (lr2[0] - ul2[0]) * (lr2[1] - ul2[1])

    return float(area_int) / (area_1 + area_2 - area_int)


def black_area_IOU(bb1, bb2, image):
    '''
    intersection over union between two bounding boxes
    '''
    lr1 = bb1['lr']
    ul1 = bb1['ul']
    lr2 = bb2['lr']
    ul2 = bb2['ul']

    new_ul = (max(ul1[0], ul2[0]), max(ul1[1], ul2[1]))
    new_lr = (min(lr1[0], lr2[0]), min(lr1[1], lr2[1]))

    bb1_subimage = image.subimage(ul1, lr1)
    bb2_subimage = image.subimage(ul2, lr2)
    intersect_subimage = image.subimage(new_ul, new_lr)

    bb1_black = bb1_subimage.black_area()[0]
    bb2_black = bb2_subimage.black_area()[0]
    intersect_black = intersect_subimage.black_area()[0]

    return float(intersect_black) / (bb1_black + bb2_black - intersect_black)


def evaluate_alignment(manuscript, ind):

    fname = '{}_{:0>3}'.format(manuscript, ind)
    with open('./ground-truth-alignments/{}_gt.json'.format(fname), 'r') as j:
        gt_boxes = json.load(j)['syl_boxes']

    with open('./out_json/{}.json'.format(fname), 'r') as j:
        align_boxes = json.load(j)['syl_boxes']

    raw_image = gc.load_image('./png/' + fname + '_text.png')
    image, _, _ = preproc.preprocess_images(raw_image, correct_rotation=False)

    score = {}
    area_score = {}
    for box in gt_boxes:
        same_syl_boxes = [x for x in align_boxes if x['syl'] == box['syl']]
        if not same_syl_boxes:
            continue
        ints = [intersect(box, x) for x in same_syl_boxes]
        if not any(ints):
            continue
        best_box = same_syl_boxes[ints.index(max(ints))]
        score[box['syl']] = IOU(box, best_box)
        area_score[box['syl']] = black_area_IOU(box, best_box, image)

    print(np.mean(score.values()), np.mean(area_score.values()))
