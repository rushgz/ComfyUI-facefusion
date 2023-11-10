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
				image_quality = gr.Slider(100, 100, 1, step=1, label="Image quality")
		return [
			img,
			enable,
			image_quality
		]

	def process(
		self,
		p: StableDiffusionProcessing,
		img,
		enable,
		image_quality
	):
		self.source = img
		self.enable = enable
		self.image_quality = image_quality
		if self.enable:
			if self.source is not None:
				# if isinstance(p, StableDiffusionProcessingImg2Img) and swap_in_source:
				# 	logger.info(f"roop enabled, face index %s", self.faces_index)
				#
				# 	for i in range(len(p.init_images)):
				# 		logger.info(f"Swap in source %s", i)
				# 		result = swap_face(
				# 			self.source,
				# 			p.init_images[i],
				# 			faces_index=self.faces_index,
				# 			model=self.model,
				# 			upscale_options=self.upscale_options,
				# 		)
				# 		p.init_images[i] = result.image()
				pass
			else:
				logger.error(f"Please provide a source face")

	def postprocess_batch(self, *args, **kwargs):
		if self.enable:
			return images

	def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
		if self.enable:
			if self.source is not None:
				image: Image.Image = script_pp.image
				result: ImageResult = swap_face(
					self.source,
					image,
					image_quality=self.image_quality,
				)
				pp = scripts_postprocessing.PostprocessedImage(result.image())
				pp.info = {}
				p.extra_generation_params.update(pp.info)
				script_pp.image = pp.image
