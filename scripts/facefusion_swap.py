# coding=utf-8

import gradio as gr
from PIL import Image
from modules import scripts, images, scripts_postprocessing
from modules.processing import (
	StableDiffusionProcessing,
)

from scripts.facefusion_logging import logger
from scripts.fusion_swapper import swap_face, ImageResult
from scripts.facefusion_utils import get_timestamp


class FaceFusionScript(scripts.Script):
	def title(self):
		return f"FaceFusion"

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		with gr.Accordion(f"FaceFusion", open=False):
			with gr.Column():
				with gr.Row():
					img = gr.Image(type="pil", label="Single Source Image")
					imgs = gr.Files(label="Multiple Source Images", file_types=["image"])
				enable = gr.Checkbox(False, placeholder="enable", label="Enable")
				device = gr.Radio(
					label="Execution Provider",
					choices=["cpu", "cuda"],
					value="cpu",
					type="value",
					scale=2
				)
				face_detector_score = gr.Slider(
					label="Face Detector Score",
					value=0.72,
					step=0.02,
					minimum=0,
					maximum=1
				)
		return [
			img,
			enable,
			device,
			face_detector_score,
			imgs
		]

	def process(
		self,
		p: StableDiffusionProcessing,
		img,
		enable,
		device,
		face_detector_score,
		imgs
	):
		self.source = img
		self.enable = enable
		self.device = device
		self.face_detector_score = face_detector_score
		self.source_imgs = imgs
		if self.enable:
			if self.source is None:
				logger.error(f"Please provide a source face")

	def postprocess_batch(self, *args, **kwargs):
		if self.enable:
			return images

	def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
		if self.enable:
			if self.source is not None:
				st = get_timestamp()
				logger.info("FaceFusion enabled, start process")
				image: Image.Image = script_pp.image
				result: ImageResult = swap_face(
					self.source,
					image,
					self.device,
					self.face_detector_score,
					self.source_imgs
				)
				pp = scripts_postprocessing.PostprocessedImage(result.image())
				pp.info = {}
				p.extra_generation_params.update(pp.info)
				script_pp.image = pp.image
				et = get_timestamp()
				cost_time = (et - st) / 1000
				logger.info(f"FaceFusion process done, time taken: {cost_time} sec.")
