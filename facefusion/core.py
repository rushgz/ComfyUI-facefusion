import os

os.environ['OMP_NUM_THREADS'] = '1'

import sys
import warnings
import platform
import shutil
import onnxruntime

import facefusion.choices
import facefusion.globals
from facefusion.face_analyser import get_one_face
from facefusion.face_reference import get_face_reference, set_face_reference
from facefusion.vision import read_image
from facefusion import wording
from facefusion.processors.frame.core import get_frame_processors_modules
from facefusion.utilities import is_image, update_status

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')


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
		update_status(wording.get('python_not_supported').format(version = '3.9'))
		return False
	return True


def conditional_process() -> None:
	conditional_set_face_reference()
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		if not frame_processor_module.pre_process('output'):
			return
	if is_image(facefusion.globals.target_path):
		process_image()


def conditional_set_face_reference() -> None:
	if 'reference' in facefusion.globals.face_selector_mode and not get_face_reference():
		if is_image(facefusion.globals.target_path):
			reference_frame = read_image(facefusion.globals.target_path)
			reference_face = get_one_face(reference_frame, facefusion.globals.reference_face_position)
			set_face_reference(reference_face)


def process_image() -> None:
	shutil.copy2(facefusion.globals.target_path, facefusion.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(facefusion.globals.frame_processors):
		update_status(wording.get('processing'), frame_processor_module.NAME)
		frame_processor_module.process_image(facefusion.globals.source_path, facefusion.globals.output_path, facefusion.globals.output_path)
		frame_processor_module.post_process()
	# validate image
	if is_image(facefusion.globals.output_path):
		update_status(wording.get('processing_image_succeed'))
	else:
		update_status(wording.get('processing_image_failed'))
