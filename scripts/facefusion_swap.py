# coding=utf-8

import gradio as gr
from PIL import Image
from modules import scripts, images, scripts_postprocessing
from modules.processing import (
	StableDiffusionProcessing,
)

from scripts.facefusion_logging import logger
from scripts.fusion_swapper import swap_face, ImageResult


class FaceFusionScript(scripts.Script):
	def title(self):
		return f"FaceFusion"

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		with gr.Accordion(f"FaceFusion", open=False):
			with gr.Column():
				img = gr.inputs.Image(type="pil")
				enable = gr.Checkbox(False, placeholder="enable", label="Enable")
		return [
			img,
			enable,
		]

	def process(
		self,
		p: StableDiffusionProcessing,
		img,
		enable,
	):
		self.source = img
		self.enable = enable
		if self.enable:
			if self.source is None:
				logger.error(f"Please provide a source face")

	def postprocess_batch(self, *args, **kwargs):
		if self.enable:
			return images

	def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
		if self.enable:
			if self.source is not None:
				logger.info("FaceFusion enabled, start process")
				image: Image.Image = script_pp.image
				result: ImageResult = swap_face(
					self.source,
					image,
				)
				pp = scripts_postprocessing.PostprocessedImage(result.image())
				pp.info = {}
				p.extra_generation_params.update(pp.info)
				script_pp.image = pp.image
				logger.info("FaceFusion process done")
