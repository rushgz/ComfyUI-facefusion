import subprocess
import pytest

import facefusion.globals
from facefusion.utilities import  conditional_download


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
	pass
