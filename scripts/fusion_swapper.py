# coding=utf-8
import os
import tempfile
from dataclasses import dataclass
from typing import Union, List

from PIL import Image

from facefusion.core import run


@dataclass
class ImageResult:
	path: Union[str, None] = None

	def image(self) -> Union[Image.Image, None]:
		if self.path:
			return Image.open(self.path)
		return None


def get_images_from_list(imgs: Union[List, None]):
	result = []
	if imgs is None:
		return result
	for x in imgs:
		import base64, io
		if 'base64,' in x:  # check if the base64 string has a data URL scheme
			base64_data = x.split('base64,')[-1]
			img_bytes = base64.b64decode(base64_data)
			source_img = Image.open(io.BytesIO(img_bytes))
			source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
			source_img.save(source_path)
			path = source_path
		else:
			path = os.path.abspath(x.name)
		result.append(path)
	return result


def swap_face(
	source_img: Image.Image,
	target_img: Image.Image,
	provider: str,
	detector_score: float,
	source_imgs: Union[List, None] = None
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
	paths = [source_path, *get_images_from_list(source_imgs)]
	result = run(paths, target_path, output_path, provider=provider, detector_score=detector_score)
	if result:
		return ImageResult(path=result)
	return ImageResult(path=target_path)
