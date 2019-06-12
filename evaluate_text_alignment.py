import PIL
import pickle
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np


def draw_ground_truth_alignment(ind):
    ind = 77
    fname = 'salzinnes_{:0>3}'.format(ind)

    with open('./ground-truth-alignments/gt_{}.json'.format(fname), 'r') as j:
        align_dict = json.load(j)

    syl_boxes = align_dict['syl_boxes']
    im = Image.open('./png/{}_text.png'.format(fname))
    text_size = 70
    fnt = ImageFont.truetype('FreeMono.ttf', text_size)
    draw = ImageDraw.Draw(im)

    for i, cbox in enumerate(syl_boxes):

        ul = tuple(cbox['ul'])
        lr = tuple(cbox['lr'])
        draw.text((ul[0], ul[1] - text_size), cbox['syl'], font=fnt, fill='black')
        draw.rectangle([ul, lr], outline='black')
        draw.line([ul[0], ul[1], ul[0], lr[1]], fill='black', width=10)

    im.save('./ground-truth-alignments/gt_{}.png'.format(fname))


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


def evaluate_alignment(ind):

    fname = 'salzinnes_{:0>3}'.format(ind)
    with open('./ground-truth-alignments/gt_{}.json'.format(fname), 'r') as j:
        gt_boxes = json.load(j)['syl_boxes']

    with open('./out_json/{}.json'.format(fname), 'r') as j:
        align_boxes = json.load(j)['syl_boxes']

    score = {}
    black_area_score = 0
    for box in gt_boxes:
        same_syl_boxes = [x for x in align_boxes if x['syl'] == box['syl']]
        if not same_syl_boxes:
            continue
        ints = [intersect(box, x) for x in same_syl_boxes]
        if not any(ints):
            continue
        best_box = same_syl_boxes[ints.index(max(ints))]
        score[box['syl']] = IOU(box, best_box)
    print(np.mean(score.values()))
