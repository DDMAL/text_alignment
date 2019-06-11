# -*- coding: utf-8 -*-


import re

consonant_groups = ['qu', 'fl', 'fr', 'st', 'br', 'cr', 'pr', 'tr', 'ct', 'th']
dipthongs = ['au', 'io', 'ihe', 'oe', 'ua', 'ui', 'uo']
vowels = ['a', 'e', 'i', 'o', 'u']

abbreviations = {
    u'dns': ['do', 'mi', 'nus'],
    u'dūs': ['do', 'mi', 'nus'],
    u'dne': ['do', 'mi', 'ne'],
    u'alla': ['al', 'le', 'lu', 'ia'],
    u'^': ['us'],
    u'ā': ['am'],
    u'ē': ['em'],
    u'ū': ['um'],
    u'ō': ['om']
}


def syllabify_word(word):
    res = word

    if res == 'euouae':
        return 'e-u-o-u-ae'

    # put square brackets around all dipthongs and vowels (using lookbehind/lookahead regex to make sure that vowels are not found within dipthongs)
    for dt in dipthongs:
        res = res.replace(dt, '[' + dt + ']')

    for v in vowels:
        regex = r"(?<!\[)" + v + r"(?!\])"
        res = re.sub(regex, '[' + v + ']', res)

    vowel_positions = [m.start() for m in re.finditer('\]', res)]
    dashes_added = 0

    for vp in vowel_positions:
        new_vp = vp + dashes_added
        next_pos = res.find('[', new_vp)

        if next_pos == -1:
            break

        interval = res[new_vp+1:next_pos]

        # if the interval has length 1 or 0, add a dash immediately after the vowel
        if len(interval) <= 1:
            res = res[:new_vp+1] + '-' + res[new_vp+1:]
            dashes_added += 1
            continue

        # for two consonants between a vowel: if they're a consonant group, assign them both to the later syllable, if not, split them down the middle
        if len(interval) == 2:
            if interval in consonant_groups:
                res = res[:new_vp+1] + '-' + res[new_vp+1:]
            else:
                res = res[:new_vp+2] + '-' + res[new_vp+2:]

            dashes_added += 1
            continue

        # check if the first two characters are a consonant group; if they are, put a dash after them, and if they're not, put a dash just after the first one
        if len(interval) >= 3:
            if interval[:2] in consonant_groups:
                res = res[:new_vp+3] + '-' + res[new_vp+3:]
            else:
                res = res[:new_vp+2] + '-' + res[new_vp+2:]

            dashes_added += 1
            continue

    # remove brackets
    res = re.sub(r"[\[\]]", "", res)

    return res


def syllabify_text(input):
    words = input.split(' ')
    word_syls = [syllabify_word(w) for w in words]
    syls = []
    for ws in word_syls:
        syls += ws.split('-')
    return syls


def parse_transcript(filename, syllabify=True):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    lines = [x for x in lines if not x[0] == '#']
    # lines = ['*' + x[1:] for x in lines]
    # lines = ' '.join(lines)

    text = ''

    for line in lines:
        line = line.lower()
        line = line.replace('|', '')
        line = line.replace('.', '')
        line = line.strip(' \t\n\r')
        words = [syllabify_word(x) for x in re.compile(' ').split(line)]
        # words[0] = '*' + words[0][1:]
        text = text + ' '.join(words) + ' '

    text = text.strip(' \t\n\r')
    text = text.replace(' ', '- ')
    text = re.compile('[-]').split(text)

    # remove empty strings, if they're in there for some reason
    text = [x for x in text if not x.isspace()]
    words_begin = []

    for i, x in enumerate(text):
        if x[0] == ' ':
            text[i] = text[i][1:]
            words_begin.append(1)
        else:
            words_begin.append(0)

    return text, words_begin


if __name__ == "__main__":
    print(parse_transcript('./png/salzinnes_18_transcript.txt'))
