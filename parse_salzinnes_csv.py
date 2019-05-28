import csv


def filename_to_text_func(transcript_path='123723_Salzinnes.csv', mapping_path='mapping.csv'):
    '''
    returns a function that, when given the filename of a salzinnes image, returns the lyrics
    on that image. to be safe, this will include chants that may partially appear on the previous
    or next page.
    '''

    mapping = []
    with open(mapping_path) as file:
        reader = csv.reader(file, delimiter=',')
        header = reader.next()
        for row in reader:
            line = {}
            line['seq'] = int(row[0])
            line['folio'] = row[1]
            line['filename'] = row[2]
            mapping.append(line)

    arr = []
    with open(transcript_path) as file:
        reader = csv.reader(file, delimiter=',')
        header = reader.next()
        for row in reader:
            arr.append(row)

    # throw away chants with no associated melody on the page (Mode == *)
    arr = [x for x in arr if not x[10] == '*']

    folio_to_chants = {}
    folio_names = set([x[2] for x in arr])              # x[2] = folio name containing chant

    for name in folio_names:

        chant_rows = [x for x in arr if x[2] == name]
        chant_rows.sort(key=lambda x: int(x[3]))        # x[3] = sequence of chants on folio

        chants = [x[14] for x in chant_rows]            # x[14] = text of chant
        folio_to_chants[name] = chants

    def fname_to_text(inp):
        entry = [x for x in mapping if inp == x['filename']]

        if not entry:
            raise ValueError('filename {} not found'.format(inp))

        entry = entry[0]

        folio = entry['folio']
        prev_entry = mapping[entry['seq'] - 1]
        prev_folio = entry['folio']

        text = folio_to_chants[prev_folio][-1]
        for chant in folio_to_chants[folio]:
            text = text + ' ' + chant

        return text

    return fname_to_text

if __name__ == '__main__':
    text_func = filename_to_text_func()
    transcript = text_func('CF-118')
    print(transcript)
