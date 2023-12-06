from typing import Optional
from functools import lru_cache
import cv2

from facefusion.typing import Frame


def resize_frame_dimension(frame : Frame, max_width : int, max_height : int) -> Frame:
	height, width = frame.shape[:2]
	if height > max_height or width > max_width:
		scale = min(max_height / height, max_width / width)
		new_width = int(width * scale)
		new_height = int(height * scale)
		return cv2.resize(frame, (new_width, new_height))
	return frame


@lru_cache(maxsize = 128)
def read_static_image(image_path : str) -> Optional[Frame]:
	return read_image(image_path)


def read_image(image_path : str) -> Optional[Frame]:
	if image_path:
		return cv2.imread(image_path)
	return None


def write_image(image_path : str, frame : Frame) -> bool:
	if image_path:
		return cv2.imwrite(image_path, frame)
	return False
