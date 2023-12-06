# coding=utf-8
import tempfile
from PIL import Image
from dataclasses import dataclass
from typing import Union

import facefusion.globals
from facefusion.core import conditional_process, limit_resources, pre_check
from facefusion import face_analyser
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.utilities import decode_execution_providers
from facefusion.utilities import normalize_output_path


@dataclass
class ImageResult:
	path: Union[str, None] = None

	def image(self) -> Union[Image.Image, None]:
		if self.path:
			return Image.open(self.path)
		return None


def apply_args(source_path, target_path, output_path, image_quality=100) -> None:
	# general
	facefusion.globals.source_path = source_path
	facefusion.globals.target_path = target_path
	facefusion.globals.output_path = normalize_output_path(facefusion.globals.source_path,
														   facefusion.globals.target_path, output_path)
	# misc
	facefusion.globals.skip_download = False
	# execution
	facefusion.globals.execution_providers = decode_execution_providers(['cuda', 'cpu'])
	facefusion.globals.execution_thread_count = 1
	facefusion.globals.execution_queue_count = 1
	facefusion.globals.max_memory = None
	# face analyser
	facefusion.globals.face_analyser_order = 'large-small'
	facefusion.globals.face_analyser_age = None
	facefusion.globals.face_analyser_gender = None
	facefusion.globals.face_detector_model = 'retinaface'
	facefusion.globals.face_detector_size = '640x640'
	facefusion.globals.face_detector_score = 0.75
	# face selector
	facefusion.globals.face_selector_mode = 'one'
	facefusion.globals.reference_face_position = 0
	facefusion.globals.reference_face_distance = 0.6
	facefusion.globals.reference_frame_number = 0
	# face mask
	facefusion.globals.face_mask_blur = 0.3
	facefusion.globals.face_mask_padding = (0, 0, 0, 0)
	# output creation
	facefusion.globals.output_image_quality = image_quality
	# frame processors
	facefusion.globals.frame_processors = ['face_swapper', 'face_enhancer']
	frame_processors_globals.face_swapper_model = "inswapper_128"
	facefusion.globals.face_recognizer_model = 'arcface_inswapper'
	frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'
	frame_processors_globals.face_enhancer_blend = 100


def swap_face(
	source_img: Image.Image,
	target_img: Image.Image,
) -> ImageResult:
	if isinstance(source_img, str):  # source_img is a base64 string
		import base64, io
		if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
			base64_data = source_img.split('base64,')[-1]
			img_bytes = base64.b64decode(base64_data)
		else:
			# if no data URL scheme, just decode
			img_bytes = base64.b64decode(source_img)
		source_img = Image.open(io.BytesIO(img_bytes))
	source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
	source_img.save(source_path)
	target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
	target_img.save(target_path)
	output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

	# call FaceFusion
	apply_args(source_path, target_path, output_path)
	limit_resources()
	if not pre_check() or not face_analyser.pre_check():
		return ImageResult()
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.pre_check():
			return ImageResult()
	conditional_process()
	return ImageResult(path=facefusion.globals.output_path)
