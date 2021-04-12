from rodan.jobs.base import RodanTask
import json
from celery.utils.log import get_task_logger
from . import align_to_ocr as align
from skimage import io
import os


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
        raw_image = io.imread(inputs['Text Layer'][0]['resource_path'])
        model_path = inputs['OCR Model'][0]['resource_path']

        # forgive me for this hack, but calamari v1.0 requires the .json extension. we should
        # upgrade to python 3.7 when we can so this is not necessary

        self.logger.info('processing image...')
        model_path_rename = inputs['OCR Model'][0]['resource_path'] + '.json'
        os.rename(model_path, model_path_rename)
        result = align.process(raw_image, transcript, model_path_rename)
        os.rename(model_path_rename, model_path)

        syl_boxes, _, lines_peak_locs, _ = result

        self.logger.info('writing output to json...')
        outfile_path = outputs['Text Alignment JSON'][0]['resource_path']
        with open(outfile_path, 'w') as file:
            json.dump(align.to_JSON_dict(syl_boxes, lines_peak_locs), file)

        return True
