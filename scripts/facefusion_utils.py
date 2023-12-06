# coding=utf-8
import time


def get_timestamp() -> int:
	"""
	获取毫秒时间戳
	:return: 毫秒
	"""
	t = time.time()
	return int(round(t * 1000))
