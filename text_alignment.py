from rodan.jobs.base import RodanTask
import gamera.core as gc
import json
import alignToOCR as align
from celery.utils.log import get_task_logger


class text_alignment(RodanTask):
    name = 'Text Alignment'
    author = 'Timothy de Reuse'
    description = 'Given a text layer image and plaintext of some text on that page, finds the'
    enabled = True
    category = 'text'
    interactive = False
    logger = get_task_logger(__name__)

    settings = {
        'title': 'Text Alignment Settings',
        'type': 'object',
        'job_queue': 'Python3',
    }

    input_port_types = [{
        'name': 'Text Layer',
        'resource_types': ['image/rgb+png'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }, {
        'name': 'Transcript',
        'resource_types': ['text/plain'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }, {
        'name': 'OCR Model',
        'resource_types': ['application/ocropus+pyrnn'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }
    ]

    output_port_types = [{
        'name': 'Text Alignment JSON',
        'resource_types': ['application/json'],
        'minimum': 1,
        'maximum': 1,
        'is_list': False
    }]

    def run_my_task(self, inputs, settings, outputs):
        self.logger.info(settings)

        transcript = align.read_file(inputs['Transcript'][0]['resource_path'])
        raw_image = gc.load_image(inputs['Text Layer'][0]['resource_path'])
        model_path = inputs['OCR Model'][0]['resource_path']

        self.logger.info('processing image...')
        id = 'wkdir'
        result = align.process(raw_image, transcript, model_path,
            wkdir_name='ocr_{}'.format(id))
        syl_boxes, _, lines_peak_locs, _ = result

        self.logger.info('writing output to json...')
        outfile_path = outputs['Text Alignment JSON'][0]['resource_path']
        with open(outfile_path, 'w') as file:
            json.dump(align.to_JSON_dict(syl_boxes, lines_peak_locs), file)

        return True
