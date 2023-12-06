# coding=utf-8
import os
import time

import scripts.facefusion_globals as gl


def get_timestamp() -> int:
	"""
	获取毫秒时间戳
	:return: 毫秒
	"""
	t = time.time()
	return int(round(t * 1000))


def set_device(value):
	gl.device_type = value
	with open(os.path.join(gl.BASE_PATH, "last_device.txt"), "w") as txt:
		txt.write(value)


def get_device():
	if not gl.is_first_run:
		return gl.device_type
	try:
		last_device_log = os.path.join(gl.BASE_PATH, "last_device.txt")
		with open(last_device_log) as f:
			last_device = f.readline().strip()
	except:
		last_device = "cpu"
	gl.device_type = last_device
	gl.is_first_run = False
	return last_device
