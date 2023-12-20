import os

os.environ['OMP_NUM_THREADS'] = '1'

import ssl
import sys
import warnings
import platform
import shutil
import onnxruntime

import facefusion.choices
import facefusion.globals
from facefusion.face_analyser import get_one_face, get_average_face
from facefusion.face_store import get_reference_faces, append_reference_face
from facefusion.vision import read_image, read_static_images
from facefusion import face_analyser, face_masker, logger, wording
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.execution_helper import decode_execution_providers
from facefusion.normalizer import normalize_output_path
from facefusion.filesystem import is_image
import facefusion.globals
from facefusion.processors.frame import globals as frame_processors_globals

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')


if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def apply_args(source_path, target_path, output_path, provider, detector_score) -> None:
	# general
	facefusion.globals.source_paths = source_path
	facefusion.globals.target_path = target_path
	facefusion.globals.output_path = normalize_output_path(
		facefusion.globals.source_paths,
		facefusion.globals.target_path,
		output_path
	)
	# misc
	facefusion.globals.skip_download = False
	facefusion.globals.log_level = 'info'
	# execution
	facefusion.globals.current_device = provider
	providers = decode_execution_providers([provider])
	if len(providers) == 0:
		providers = decode_execution_providers(['cpu'])
	facefusion.globals.execution_providers = providers
	logger.info(f"device use {facefusion.globals.execution_providers}", __name__.upper())
	facefusion.globals.execution_thread_count = 1
	facefusion.globals.execution_queue_count = 1
	facefusion.globals.max_memory = None
	# face analyser
	facefusion.globals.face_analyser_order = 'large-small'
	facefusion.globals.face_analyser_age = None
	facefusion.globals.face_analyser_gender = None
	facefusion.globals.face_detector_model = 'retinaface'
	facefusion.globals.face_detector_size = '640x640'
	facefusion.globals.face_detector_score = detector_score
	# face selector
	facefusion.globals.face_selector_mode = 'one'
	facefusion.globals.reference_face_position = 0
	facefusion.globals.reference_face_distance = 0.6
	facefusion.globals.reference_frame_number = 0
	# face mask
	facefusion.globals.face_mask_types = ['box']
	facefusion.globals.face_mask_blur = 0.3
	facefusion.globals.face_mask_padding = (0, 0, 0, 0)
	facefusion.globals.face_mask_regions = facefusion.choices.face_mask_regions
	# output creation
	facefusion.globals.output_image_quality = 100
	# frame processors
	facefusion.globals.frame_processors = ['face_swapper', 'face_enhancer']
	frame_processors_globals.face_swapper_model = "inswapper_128"
	facefusion.globals.face_recognizer_model = 'arcface_inswapper'
	frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'
	frame_processors_globals.face_enhancer_blend = 100


def run(source_path, target_path, output_path, provider="cpu", detector_score=0.72):
	apply_args(source_path, target_path, output_path, provider, detector_score)
	logger.init(facefusion.globals.log_level)
	limit_resources()
	if not pre_check() or not face_analyser.pre_check() or not face_masker.pre_check():
		return None
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.pre_check():
			return None
	conditional_process()
	if is_image(output_path):
		return facefusion.globals.output_path
	return None


def limit_resources() -> None:
	if facefusion.globals.max_memory:
		memory = facefusion.globals.max_memory * 1024 ** 3
		if platform.system().lower() == 'darwin':
			memory = facefusion.globals.max_memory * 1024 ** 6
		if platform.system().lower() == 'windows':
			import ctypes

			kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		else:
			import resource

			resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
	if sys.version_info < (3, 9):
		logger.error(wording.get('python_not_supported').format(version = '3.9'), __name__.upper())
		return False
	return True


def conditional_process() -> None:
	conditional_append_reference_faces()
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.pre_process('output'):
			return
	if is_image(facefusion.globals.target_path):
		process_image()


def conditional_append_reference_faces() -> None:
	if 'reference' in facefusion.globals.face_selector_mode and not get_reference_faces():
		source_frames = read_static_images(facefusion.globals.source_paths)
		source_face = get_average_face(source_frames)
		if is_image(facefusion.globals.target_path):
			reference_frame = read_image(facefusion.globals.target_path)
			reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
			append_reference_face('origin', reference_face)
			if source_face and reference_face:
				for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
					reference_frame = frame_processor_module.get_reference_frame(source_face, reference_face,
																				 reference_frame)
					reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
					append_reference_face(frame_processor_module.__name__, reference_face)


def process_image() -> None:
	shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		logger.info(wording.get('processing'), frame_processor_module.NAME)
		logger.info(f'current device: {facefusion.globals.current_device}, last device: {facefusion.globals.last_device}', frame_processor_module.NAME)
		if facefusion.globals.current_device != facefusion.globals.last_device:
			logger.info('device changed, post models', frame_processor_module.NAME)
			frame_processor_module.post_models()
		facefusion.globals.last_device = facefusion.globals.current_device
		frame_processor_module.process_image(facefusion.globals.source_paths, facefusion.globals.output_path,
											 facefusion.globals.output_path)
		frame_processor_module.post_process()
	# validate image
	if is_image(facefusion.globals.output_path):
		logger.info(wording.get('processing_image_succeed'), __name__.upper())
	else:
		logger.error(wording.get('processing_image_failed'), __name__.upper())
