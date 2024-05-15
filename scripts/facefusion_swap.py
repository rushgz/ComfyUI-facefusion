# coding=utf-8

import gradio as gr
from PIL import Image
from modules import scripts, images, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
)

import scripts.facefusion_logging as logger
from scripts.fusion_swapper import swap_face
from scripts.facefusion_utils import get_timestamp
import facefusion.metadata as ff_metadata

print(
    f"[-] FaceFusion initialized. version: {ff_metadata.get('version')}"
)


class FaceFusionScript(scripts.Script):
    def title(self):
        return f"FaceFusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f"FaceFusion", open=False):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(value=f"v{ff_metadata.get('version')}")
                with gr.Row():
                    img = gr.Image(type="pil", label="Single Source Image")
                    imgs = gr.Files(label="Multiple Source Images", file_types=["image"])
                with gr.Row():
                    enable = gr.Checkbox(False, placeholder="enable", label="Enable")
                    skip_nsfw = gr.Checkbox(True, placeholder="skip_nsfw", label="Skip Check NSFW")
                device = gr.Radio(
                    label="Execution Provider",
                    choices=["cpu", "cuda"],
                    value="cpu",
                    type="value",
                    scale=2
                )
                face_detector_score = gr.Slider(
                    label="Face Detector Score",
                    value=0.65,
                    step=0.02,
                    minimum=0,
                    maximum=1
                )
                mask_blur = gr.Slider(
                    label="Face Mask Blur",
                    value=0.7,
                    step=0.05,
                    minimum=0,
                    maximum=1
                )
                landmarker_score = gr.Slider(
                    label="Face Landmarker Score",
                    value=0.5,
                    step=0.05,
                    minimum=0,
                    maximum=1
                )
        return [
            img,
            enable,
            device,
            face_detector_score,
            mask_blur,
            imgs,
            skip_nsfw,
            landmarker_score
        ]

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        device,
        face_detector_score,
        mask_blur,
        imgs,
        skip_nsfw,
        landmarker_score
    ):
        self.source = img
        self.enable = enable
        self.device = device
        self.face_detector_score = face_detector_score
        self.mask_blur = mask_blur
        self.source_imgs = imgs
        self.skip_nsfw = skip_nsfw
        self.landmarker_score = landmarker_score
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
                landmarker_score = 0.5
                if self.landmarker_score:
                    landmarker_score = self.landmarker_score
                result: Image.Image = swap_face(
                    self.source,
                    image,
                    self.device,
                    self.face_detector_score,
                    self.mask_blur,
                    landmarker_score,
                    self.skip_nsfw,
                    self.source_imgs
                )
                pp = scripts_postprocessing.PostprocessedImage(result)
                pp.info = {}
                p.extra_generation_params.update(pp.info)
                script_pp.image = pp.image
                et = get_timestamp()
                cost_time = (et - st) / 1000
                logger.info(f"FaceFusion process done, time taken: {cost_time} sec.")
