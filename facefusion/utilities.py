from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
import filetype
import os
import platform
import shutil
import ssl
import subprocess
import tempfile
import urllib.request
import onnxruntime

import facefusion.globals
from facefusion import wording
from facefusion.typing import Padding

TEMP_DIRECTORY_PATH = os.path.join(tempfile.gettempdir(), 'facefusion')
TEMP_OUTPUT_VIDEO_NAME = 'temp.mp4'

# monkey patch ssl
if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def get_temp_directory_path(target_path : str) -> str:
	target_name, _ = os.path.splitext(os.path.basename(target_path))
	return os.path.join(TEMP_DIRECTORY_PATH, target_name)


def clear_temp(target_path : str) -> None:
	temp_directory_path = get_temp_directory_path(target_path)
	parent_directory_path = os.path.dirname(temp_directory_path)
	if is_directory(temp_directory_path):
		shutil.rmtree(temp_directory_path)
	if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
		os.rmdir(parent_directory_path)


def normalize_output_path(source_path : Optional[str], target_path : Optional[str], output_path : Optional[str]) -> Optional[str]:
	if is_file(target_path) and is_directory(output_path):
		target_name, target_extension = os.path.splitext(os.path.basename(target_path))
		if is_file(source_path):
			source_name, _ = os.path.splitext(os.path.basename(source_path))
			return os.path.join(output_path, source_name + '-' + target_name + target_extension)
		return os.path.join(output_path, target_name + target_extension)
	if is_file(target_path) and output_path:
		_, target_extension = os.path.splitext(os.path.basename(target_path))
		output_name, output_extension = os.path.splitext(os.path.basename(output_path))
		output_directory_path = os.path.dirname(output_path)
		if is_directory(output_directory_path) and output_extension:
			return os.path.join(output_directory_path, output_name + target_extension)
		return None
	return output_path


def normalize_padding(padding : Optional[List[int]]) -> Optional[Padding]:
	if padding and len(padding) == 1:
		return tuple([ padding[0], padding[0], padding[0], padding[0] ]) # type: ignore[return-value]
	if padding and len(padding) == 2:
		return tuple([ padding[0], padding[1], padding[0], padding[1] ]) # type: ignore[return-value]
	if padding and len(padding) == 3:
		return tuple([ padding[0], padding[1], padding[2], padding[1] ]) # type: ignore[return-value]
	if padding and len(padding) == 4:
		return tuple(padding) # type: ignore[return-value]
	return None


def is_file(file_path : str) -> bool:
	return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path : str) -> bool:
	return bool(directory_path and os.path.isdir(directory_path))


def is_image(image_path : str) -> bool:
	if is_file(image_path):
		mimetype = filetype.guess(image_path).mime
		return bool(mimetype and mimetype.startswith('image/'))
	return False


def is_video(video_path : str) -> bool:
	if is_file(video_path):
		mimetype = filetype.guess(video_path).mime
		return bool(mimetype and mimetype.startswith('video/'))
	return False


def conditional_download(download_directory_path : str, urls : List[str]) -> None:
	with ThreadPoolExecutor() as executor:
		for url in urls:
			executor.submit(get_download_size, url)
	for url in urls:
		download_file_path = os.path.join(download_directory_path, os.path.basename(url))
		total = get_download_size(url)
		if is_file(download_file_path):
			initial = os.path.getsize(download_file_path)
		else:
			initial = 0
		if initial < total:
			with tqdm(total = total, initial = initial, desc = wording.get('downloading'), unit = 'B', unit_scale = True, unit_divisor = 1024, ascii = ' =') as progress:
				subprocess.Popen([ 'curl', '--create-dirs', '--silent', '--insecure', '--location', '--continue-at', '-', '--output', download_file_path, url ])
				current = initial
				while current < total:
					if is_file(download_file_path):
						current = os.path.getsize(download_file_path)
						progress.update(current - progress.n)


@lru_cache(maxsize = None)
def get_download_size(url : str) -> int:
	try:
		response = urllib.request.urlopen(url, timeout = 10)
		return int(response.getheader('Content-Length'))
	except (OSError, ValueError):
		return 0


def is_download_done(url : str, file_path : str) -> bool:
	if is_file(file_path):
		return get_download_size(url) == os.path.getsize(file_path)
	return False


def resolve_relative_path(path : str) -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def list_module_names(path : str) -> Optional[List[str]]:
	if os.path.exists(path):
		files = os.listdir(path)
		return [ Path(file).stem for file in files if not Path(file).stem.startswith(('.', '__')) ]
	return None


def encode_execution_providers(execution_providers : List[str]) -> List[str]:
	return [ execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
	available_execution_providers = onnxruntime.get_available_providers()
	encoded_execution_providers = encode_execution_providers(available_execution_providers)
	return [ execution_provider for execution_provider, encoded_execution_provider in zip(available_execution_providers, encoded_execution_providers) if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers) ]


def map_device(execution_providers : List[str]) -> str:
	if 'CoreMLExecutionProvider' in execution_providers:
		return 'mps'
	if 'CUDAExecutionProvider' in execution_providers or 'ROCMExecutionProvider' in execution_providers :
		return 'cuda'
	if 'OpenVINOExecutionProvider' in execution_providers:
		return 'mkl'
	return 'cpu'


def create_metavar(ranges : List[Any]) -> str:
	return '[' + str(ranges[0]) + '-' + str(ranges[-1]) + ']'


def update_status(message : str, scope : str = 'FACEFUSION.CORE') -> None:
	print('[' + scope + '] ' + message)
