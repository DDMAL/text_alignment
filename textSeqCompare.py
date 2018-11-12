import numpy as np
from unidecode import unidecode
import matplotlib.pyplot as plt

def read_file(fname):
    file = open(fname, 'r', encoding='utf8')
    lines = file.readlines()
    file.close()
    lines = ' '.join(lines)
    lines = lines.replace('\n', '')
    lines = unidecode(lines)
    return lines


# scoring system
match = 4
mismatch = -4
gap_open = -4
gap_extend = -4

gap_open_x = -4
gap_extend_x = -4
gap_open_y = -4
gap_extend_y = -4

# display length
line_len = 90

# files
files = ['einsiedeln_001r', 'einsiedeln_001v', 'einsiedeln_002r',
    'einsiedeln_002v', 'einsiedeln_003r', 'einsiedeln_003v',
    'salzinnes_15', 'salzinnes_17', 'stgall390_25',
    'stmaurf_49r']


def process(item):
    print('processing ' + item + '...')
    transcript = read_file('./txt/' + item + '_transcript.txt')
    ocr = read_file('./txt/' + item + '_ocr.txt')

    # transcript = 'dafsad'
    # ocr = 'dfasd'

    # y_mat and x_mat keep track of gaps in horizontal and vertical directions
    mat = np.zeros((len(transcript), len(ocr)))
    y_mat = np.zeros((len(transcript), len(ocr)))
    x_mat = np.zeros((len(transcript), len(ocr)))
    mat_ptr = np.zeros((len(transcript), len(ocr)))
    y_mat_ptr = np.zeros((len(transcript), len(ocr)))
    x_mat_ptr = np.zeros((len(transcript), len(ocr)))

    for i in range(len(transcript)):
        mat[i][0] = gap_extend * i
        x_mat[i][0] = -100000
        y_mat[i][0] = gap_extend * i
    for j in range(len(ocr)):
        mat[0][j] = gap_extend * j
        x_mat[0][j] = gap_extend * j
        y_mat[0][j] = -100000

    for i in range(1, len(transcript)):
        for j in range(1, len(ocr)):

            # update main matrix (for matches)
            match_score = match if transcript[i-1] == ocr[j-1] else mismatch
            mat_vals = [mat[i-1][j-1], x_mat[i-1][j-1], y_mat[i-1][j-1]]
            mat[i][j] = max(mat_vals) + match_score
            mat_ptr[i][j] = int(mat_vals.index(max(mat_vals)))

            # update matrix for y gaps
            y_mat_vals = [mat[i][j-1] + gap_open_y + gap_extend_y,
                        x_mat[i][j-1] + gap_open_y + gap_extend_y,
                        y_mat[i][j-1] + gap_extend_y]

            y_mat[i][j] = max(y_mat_vals)
            y_mat_ptr[i][j] = int(y_mat_vals.index(max(y_mat_vals)))

            # update matrix for x gaps
            x_mat_vals = [mat[i-1][j] + gap_open_x + gap_extend_x,
                        x_mat[i-1][j] + gap_extend_x,
                        y_mat[i-1][j] + gap_open_x + gap_extend_x]

            x_mat[i][j] = max(x_mat_vals)
            x_mat_ptr[i][j] = int(x_mat_vals.index(max(x_mat_vals)))

    # asymetric indel?

    # TRACEBACK
    # which matrix we're in tells us which direction to head back (diagonally, y, or x)
    # value of that matrix tells us which matrix to go to (mat, y_mat, or x_mat)
    # mat of 0 = match, 1 = x gap, 2 = y gap
    #
    # first
    tra_align = ''
    ocr_align = ''
    align_record = ''
    pt_record = ''
    xpt = len(transcript) - 1
    ypt = len(ocr) - 1
    mpt = mat_ptr[xpt][ypt]
    prev_pt = -1

    # start at bottom-right corner and work way up to top-left
    while(xpt >= 0 and ypt >= 0):

        pt_record += str(int(mpt))

        # case if the current cell is reachable from the diagonal
        if mpt == 0:
            tra_align += transcript[xpt - 1]
            ocr_align += ocr[ypt - 1]
            added_text = transcript[xpt] + ' ' + ocr[ypt]
            print(mpt, xpt, ypt, added_text)

            # determine if this diagonal step was a match or a mismatch
            align_record += 'O' if(transcript[xpt - 1] == ocr[ypt - 1]) else 'X'

            mpt = mat_ptr[xpt][ypt]
            xpt -= 1
            ypt -= 1

        # case if current cell is reachable horizontally
        elif mpt == 1:
            tra_align += transcript[xpt - 1]
            ocr_align += '_'
            added_text = transcript[xpt] + ' _'
            print(mpt, xpt, ypt, added_text)

            align_record += ' '
            mpt = x_mat_ptr[xpt][ypt]
            xpt -= 1

        # case if current cell is reachable vertically
        elif mpt == 2:
            tra_align += '_'
            ocr_align += ocr[ypt - 1]
            added_text = '_ ' + ocr[ypt]
            print(mpt, xpt, ypt, added_text)

            align_record += ' '
            mpt = y_mat_ptr[xpt][ypt]
            ypt -= 1

    # reverse all records, since we obtained them by traversing the matrices from the bottom-right
    tra_align = tra_align[::-1]
    ocr_align = ocr_align[::-1]
    align_record = align_record[::-1]
    pt_record = pt_record[::-1]

    # log results
    file = open('./results/' + item + '_result.txt', 'w+')
    file.seek(0)
    file.truncate()
    for n in range(int(np.ceil(len(tra_align) / line_len))):
        start = n * line_len
        end = (n + 1) * line_len
        print(tra_align[start:end])
        print(ocr_align[start:end])
        print(align_record[start:end])
        # print(pt_record[start:end])
        print('')

        file.write(tra_align[start:end] + '\n')
        file.write(ocr_align[start:end] + '\n')
        file.write(align_record[start:end] + '\n\n')
    file.close()

    # plt.subplot(1, 3, 1)
    # plt.imshow(mat[1:200, 1:200])
    # plt.subplot(1, 3, 2)
    # plt.imshow(x_mat[1:200, 1:200])
    # plt.subplot(1, 3, 3)
    # plt.imshow(y_mat[1:200, 1:200])
    # plt.colorbar()
    # plt.show()

    return(tra_align, ocr_align)


if __name__ == '__main__':
    process('einsiedeln_001v')
    # for f in files:
    #      process(f)
