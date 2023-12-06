import glob
import platform
import subprocess
import pytest

import facefusion.globals
from facefusion.utilities import conditional_download, normalize_output_path, normalize_padding, is_file, is_directory, is_image, is_video, get_download_size, is_download_done, encode_execution_providers, decode_execution_providers


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	facefusion.globals.temp_frame_quality = 100
	facefusion.globals.trim_frame_start = None
	facefusion.globals.trim_frame_end = None
	facefusion.globals.temp_frame_format = 'png'
	conditional_download('.assets/examples',
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4'
	])


@pytest.fixture(scope = 'function', autouse = True)
def before_each() -> None:
	facefusion.globals.trim_frame_start = None
	facefusion.globals.trim_frame_end = None
	facefusion.globals.temp_frame_quality = 90
	facefusion.globals.temp_frame_format = 'jpg'


def test_normalize_output_path() -> None:
	if platform.system().lower() != 'windows':
		assert normalize_output_path('.assets/examples/source.jpg', None, '.assets/examples/target-240p.mp4') == '.assets/examples/target-240p.mp4'
		assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/examples/target-240p.mp4') == '.assets/examples/target-240p.mp4'
		assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/examples') == '.assets/examples/target-240p.mp4'
		assert normalize_output_path('.assets/examples/source.jpg', '.assets/examples/target-240p.mp4', '.assets/examples') == '.assets/examples/source-target-240p.mp4'
		assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/examples/output.mp4') == '.assets/examples/output.mp4'
		assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/output.mov') == '.assets/output.mp4'
	assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/examples/invalid') is None
	assert normalize_output_path(None, '.assets/examples/target-240p.mp4', '.assets/invalid/output.mp4') is None
	assert normalize_output_path(None, '.assets/examples/target-240p.mp4', 'invalid') is None
	assert normalize_output_path('.assets/examples/source.jpg', '.assets/examples/target-240p.mp4', None) is None


def test_normalize_padding() -> None:
	assert normalize_padding([ 0, 0, 0, 0 ]) == (0, 0, 0, 0)
	assert normalize_padding([ 1 ]) == (1, 1, 1, 1)
	assert normalize_padding([ 1, 2 ]) == (1, 2, 1, 2)
	assert normalize_padding([ 1, 2, 3 ]) == (1, 2, 3, 2)
	assert normalize_padding(None) is None


def test_is_file() -> None:
	assert is_file('.assets/examples/source.jpg') is True
	assert is_file('.assets/examples') is False
	assert is_file('invalid') is False


def test_is_directory() -> None:
	assert is_directory('.assets/examples') is True
	assert is_directory('.assets/examples/source.jpg') is False
	assert is_directory('invalid') is False


def test_is_image() -> None:
	assert is_image('.assets/examples/source.jpg') is True
	assert is_image('.assets/examples/target-240p.mp4') is False
	assert is_image('invalid') is False


def test_is_video() -> None:
	assert is_video('.assets/examples/target-240p.mp4') is True
	assert is_video('.assets/examples/source.jpg') is False
	assert is_video('invalid') is False


def test_get_download_size() -> None:
	assert get_download_size('https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4') == 191675
	assert get_download_size('https://github.com/facefusion/facefusion-assets/releases/download/examples/target-360p.mp4') == 370732
	assert get_download_size('invalid') == 0


def test_is_download_done() -> None:
	assert is_download_done('https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4', '.assets/examples/target-240p.mp4') is True
	assert is_download_done('https://github.com/facefusion/facefusion-assets/releases/download/examples/target-240p.mp4','invalid') is False
	assert is_download_done('invalid', 'invalid') is False


def test_encode_execution_providers() -> None:
	assert encode_execution_providers([ 'CPUExecutionProvider' ]) == [ 'cpu' ]


def test_decode_execution_providers() -> None:
	assert decode_execution_providers([ 'cpu' ]) == [ 'CPUExecutionProvider' ]
