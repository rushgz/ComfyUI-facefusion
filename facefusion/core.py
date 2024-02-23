import os

os.environ['OMP_NUM_THREADS'] = '1'

import sys
import time
import warnings
import shutil
import numpy
import onnxruntime
from time import sleep

import facefusion.choices
import facefusion.globals
from facefusion.face_analyser import get_one_face, get_average_face
from facefusion.face_store import get_reference_faces, append_reference_face
from facefusion import face_analyser, face_masker, content_analyser, config, metadata, logger, wording
from facefusion.content_analyser import analyse_image
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.execution_helper import decode_execution_providers
from facefusion.normalizer import normalize_output_path, normalize_padding, normalize_fps
from facefusion.memory import limit_system_memory
from facefusion.filesystem import list_directory, get_temp_frame_paths, create_temp, move_temp, clear_temp, is_image, is_video
from facefusion.vision import get_video_frame, read_image, read_static_images, pack_resolution, detect_video_resolution, detect_video_fps, create_video_resolutions
from facefusion.processors.frame import globals as frame_processors_globals

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')


def apply_args(source_path, target_path, output_path, provider, detector_score, mask_blur, skip_nsfw) -> None:
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
	facefusion.globals.execution_thread_count = 4
	facefusion.globals.execution_queue_count = 1
	# memory
	facefusion.globals.video_memory_strategy = 'strict'
	facefusion.globals.system_memory_limit = 0
	# face analyser
	facefusion.globals.face_analyser_order = 'large-small'
	facefusion.globals.face_analyser_age = None
	facefusion.globals.face_analyser_gender = None
	facefusion.globals.face_detector_model = 'yoloface'
	facefusion.globals.face_detector_size = '640x640'
	facefusion.globals.face_detector_score = detector_score
	facefusion.globals.skip_nsfw = skip_nsfw
	# face selector
	facefusion.globals.face_selector_mode = 'one'
	facefusion.globals.reference_face_position = 0
	facefusion.globals.reference_face_distance = 0.6
	facefusion.globals.reference_frame_number = 0
	# face mask
	facefusion.globals.face_mask_types = facefusion.choices.face_mask_types
	facefusion.globals.face_mask_blur = mask_blur
	facefusion.globals.face_mask_padding = (0, 0, 0, 0)
	facefusion.globals.face_mask_regions = facefusion.choices.face_mask_regions
	# output creation
	facefusion.globals.output_image_quality = 100
	# frame processors
	facefusion.globals.frame_processors = ['face_swapper', 'face_enhancer']
	frame_processors_globals.face_swapper_model = "inswapper_128_fp16"
	facefusion.globals.face_recognizer_model = 'arcface_inswapper'
	frame_processors_globals.face_enhancer_model = 'gfpgan_1.4'
	frame_processors_globals.face_enhancer_blend = 100


def run(source_path, target_path, output_path, provider="cpu", detector_score=0.6, mask_blur=0.3, skip_nsfw=True):
	apply_args(source_path, target_path, output_path, provider, detector_score, mask_blur, skip_nsfw)
	if not facefusion.globals.model_path_checked:
		if facefusion.globals.system_memory_limit > 0:
			limit_system_memory(facefusion.globals.system_memory_limit)
		if not pre_check() or not face_analyser.pre_check() or not face_masker.pre_check():
			return None
		for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
			if not frame_processor_module.pre_check():
				return None
	if not skip_nsfw and not content_analyser.pre_check():
		return None
	conditional_process()
	if is_image(output_path):
		return facefusion.globals.output_path
	return None


def pre_check() -> bool:
	if sys.version_info < (3, 9):
		logger.error(wording.get('python_not_supported').format(version = '3.9'), __name__.upper())
		return False
	return True


def conditional_process() -> None:
	start_time = time.time()
	if not face_analyser.post_check() or not face_masker.post_check():
		facefusion.globals.model_path_checked = False
		return
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.post_check():
			facefusion.globals.model_path_checked = False
			return
		if not frame_processor_module.pre_process('output'):
			return
	if not facefusion.globals.skip_nsfw and not content_analyser.post_check():
		return
	facefusion.globals.model_path_checked = True
	conditional_append_reference_faces()
	if is_image(facefusion.globals.target_path):
		process_image(start_time)


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
					abstract_reference_frame = frame_processor_module.get_reference_frame(source_face, reference_face,
																						  reference_frame)
					if numpy.any(abstract_reference_frame):
						reference_frame = abstract_reference_frame
						reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
						append_reference_face(frame_processor_module.__name__, reference_face)


def process_image(start_time : float) -> None:
	if (not facefusion.globals.skip_nsfw) and analyse_image(facefusion.globals.target_path):
		logger.info(f"skip process, source image is nsfw", "CORE")
		return
	shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	need_post_models = facefusion.globals.current_device != facefusion.globals.last_device
	facefusion.globals.last_device = facefusion.globals.current_device
	logger.info(f"processor module: {facefusion.globals.frame_processors}", "CORE")
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		logger.info(wording.get('processing'), frame_processor_module.NAME)
		if need_post_models:
			logger.info('device changed, post models', frame_processor_module.NAME)
			frame_processor_module.post_models()
		processor_success = frame_processor_module.process_image(facefusion.globals.source_paths, facefusion.globals.output_path, facefusion.globals.output_path)
		frame_processor_module.post_process()
		if not processor_success:
			return

	# validate image
	if is_image(facefusion.globals.output_path):
		seconds = '{:.2f}'.format((time.time() - start_time) % 60)
		logger.info(wording.get('processing_image_succeed').format(seconds=seconds), __name__.upper())
	else:
		logger.error(wording.get('processing_image_failed'), __name__.upper())
