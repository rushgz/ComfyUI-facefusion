# coding=utf-8
from dataclasses import dataclass
from typing import Union, Dict

from PIL import Image
import tempfile

from facefusion.uis.components import output

from facefusion.core import apply_args, get_argument_parser, conditional_process, pre_check
from facefusion.processors.frame.modules import face_enhancer, face_swapper, frame_enhancer
import facefusion.globals


@dataclass
class ImageResult:
	path: Union[str, None] = None

	def image(self) -> Union[Image.Image, None]:
		if self.path:
			return Image.open(self.path)
		return None


def swap_face(
	source_img: Image.Image,
	target_img: Image.Image,
	image_quality: int = 80,
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
	apply_args(get_argument_parser())
	limit_resources()
	if not pre_check():
		return ImageResult()
	if (
		not face_enhancer.pre_check()
		or not face_swapper.pre_check()
		or not frame_enhancer.pre_check()
	):
		return ImageResult()
	source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
	source_img.save(source_path)
	target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
	target_img.save(target_path)
	facefusion.globals.source_path = source_path
	facefusion.globals.target_path = target_path
	facefusion.globals.output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
	facefusion.globals.output_image_quality = image_quality
	facefusion.globals.frame_processors = ['face_swapper', 'face_enhancer']
	conditional_process()
	return ImageResult(path=facefusion.globals.output_path)
