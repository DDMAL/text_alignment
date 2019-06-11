# -*- coding: utf-8 -*-
import re

consonant_groups = ['qu', 'ch', 'ph', 'fl', 'fr', 'st', 'br', 'cr', 'cl', 'pr', 'tr', 'ct', 'th']
diphthongs = ['ae', 'au', 'io', 'ie', 'ihe', 'oe', 'ua', 'iu']
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


def syllabify_word(inp):
    '''
    separate each word into UNITS - first isolate consonant groups, then diphthongs, then letters.
    each vowel / diphthong unit is a "seed" of a syllable; consonants and consonant groups "stick"
    to adjacent seeds. first make every vowel stick to its preceding consonant group. any remaining
    consonant groups stick to the vowel behind them.
    '''

    if inp == 'euouae':
        return 'e-u-o-u-ae'.split('-')

    word = [inp]

    for unit in consonant_groups + diphthongs:
        new_word = []
        for segment in word:
            if '*' in segment:
                new_word.append(segment)
                continue

            split = segment.split(unit)

            rep_list = [unit + '*'] * len(split)
            interleaved = [val for pair in zip(split, rep_list) for val in pair]

            # remove blanks and chop off last extra entry caused by list comprehension
            interleaved = [x for x in interleaved[:-1] if len(x) > 0]
            new_word += interleaved
        word = list(new_word)

    # now split into individual characters anything remaining
    new_word = []
    for segment in word:
        if '*' in segment:
            new_word.append(segment.replace('*', ''))
            continue
        new_word += list(segment)
    word = list(new_word)

    # add marker to units to mark vowels or diphthongs this time
    for i in range(len(word)):
        if word[i] in vowels + diphthongs:
            word[i] = word[i] + '*'

    # begin merging units together.
    while not all(('*' in x) for x in word):

        # first stick consonants / consonant groups to syllables ahead of them
        new_word = []
        i = 0
        while i < len(word):
            if i + 1 >= len(word):
                new_word.append(word[i])
                break
            cur = word[i]
            proc = word[i + 1]
            if '*' in proc and '*' not in cur:
                new_word.append(cur + proc)
                i += 2
            else:
                new_word.append(cur)
                i += 1
        word = list(new_word)

        # then stick consonants / consonant groups to syllables behind them
        new_word = []
        i = 0
        while i < len(word):
            if i + 1 >= len(word):
                new_word.append(word[i])
                break
            cur = word[i]
            proc = word[i + 1]
            if '*' in cur and '*' not in proc:
                new_word.append(cur + proc)
                i += 2
            else:
                new_word.append(cur)
                i += 1
        word = list(new_word)

    word = [x.replace('*', '') for x in new_word]

    return word


def syllabify_word_old(word):
    res = word

    if res == 'euouae':
        return 'e-u-o-u-ae'

    # put square brackets around all diphthongs and vowels (using lookbehind/lookahead regex to make sure that vowels are not found within diphthongs)
    for dt in diphthongs:
        res = res.replace(dt, '[' + dt + ']')

    for v in vowels:
        regex = r"(?<!\[)" + v + r"(?!\])"
        res = re.sub(regex, '[' + v + ']', res)

    vowel_positions = [m.start() for m in re.finditer(r'\]', res)]
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
    syls = [item for sublist in word_syls for item in sublist]
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
    # print(parse_transcript('./salzinnes_ocr/salzinnes_018_ocr.txt'))
    inp = 'quaecumque eius michi antiphonum assistens cernerent'
    res = syllabify_text(inp)
    print(res)
