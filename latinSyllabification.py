# -*- coding: utf-8 -*-
import re

consonant_groups = ['qu', 'ch', 'ph', 'fl', 'fr', 'st', 'br', 'cr', 'cl', 'pr', 'tr', 'ct', 'th', 'sp']
diphthongs = ['ae', 'au', 'ei', 'oe', 'ui', 'ya', 'ex', 'ix']
vowels = ['a', 'e', 'i', 'o', 'u', 'y']

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
    #
    if len(inp) <= 1:
        return inp
    if inp == 'euouae':
        return 'e-u-o-u-ae'.split('-')
    if inp == 'cuius':
        return 'cu-ius'.split('-')
    if inp == 'eius':
        return 'e-ius'.split('-')
    if inp == 'iugum':
        return 'iu-gum'.split('-')
    if inp == 'iustum':
        return 'iu-stum'.split('-')
    if inp == 'iusticiam':
        return 'iu-sti-ci-am'.split('-')
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


def syllabify_text(input):
    words = input.split(' ')
    word_syls = [syllabify_word(w) for w in words]
    syls = [item for sublist in word_syls for item in sublist]
    return syls


if __name__ == "__main__":
    inp = 'quaecumque ejus michi antiphonum assistens alleluya dixit extra exhibeamus s'
    res = syllabify_text(inp)
    print(res)
