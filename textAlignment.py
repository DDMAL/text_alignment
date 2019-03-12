from rodan.jobs.base import RodanTask
import gamera.core as gc
import alignToOCR as align


class textAlignment(RodanTask):
    name = 'Text Alignment'
    author = 'Timothy de Reuse'
    description = 'Given a text layer image and plaintext of some text on that page, finds the'
    enabled = True
    category = 'text'
    interactive = False

    settings = {
        'title': 'Text Alignment Settings',
        'type': 'object',
        'required': ['MEI Version'],
        'properties': {
            'MEI Version': {
                'enum': ['4.0.0', '3.9.9'],
                'type': 'string',
                'default': '3.9.9',
                'description': 'Specifies the MEI version, 3.9.9 is the old unofficial MEI standard used by Neon',
            },
        }
    }

    input_port_types = [{
        'name': 'Text Layer',
        'resource_types': ['image/rgba+png'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }, {
        'name': 'Transcript',
        'resource_types': ['text/plain'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    output_port_types = [{
        'name': 'JSON',
        'resource_types': ['application/JSON'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):

        transcript = align.read_file(inputs['Transcript'][0]['resource_path'])
        raw_image = gc.load_image(inputs['Text Layer'][0]['resource_path'])

        syls_boxes, _, lines_peak_locs = align.process(raw_image, transcript, wkdir_name='test')
        test_string = str(syls_boxes)

        outfile_path = outputs['JSON'][0]['resource_path']
        with open(outfile_path, 'w') as file:
            file.write(test_string)

        return True
