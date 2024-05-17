from scripts.facefusion_swap import FaceFusionScript
from modules.processing import (
    StableDiffusionProcessingImg2Img
)
from .utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil


class FaceFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "single_source_image": ("IMAGE",),  # Single source image
                "enable": ("BOOLEAN", {"default": True}),  # Enable processing
                "enable_swapper": ("BOOLEAN", {"default": True}),  # Enable swap face
                "enable_face_restore": ("BOOLEAN", {"default": True}),  # Enable face restore
                "skip_nsfw": ("BOOLEAN", {"default": True}),  # Skip NSFW check
                "device": (["cpu", "cuda"], {"default": "cpu"}),  # Execution provider
                "face_detector_score": ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.02}),
                # Face detector score
                "mask_blur": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),  # Face mask blur
                "landmarker_score": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05})
                # Face landmarker score
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "FaceFusion"

    def execute(self, image, single_source_image, enable, device, face_detector_score, mask_blur, skip_nsfw, landmarker_score, enable_swapper, enable_face_restore):
        pil_images = batch_tensor_to_pil(image)
        source = tensor_to_pil(single_source_image)
        script = FaceFusionScript()
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(p=p,
                       img=source,
                       enable=enable,
                       device=device,
                       face_detector_score=face_detector_score,
                       mask_blur=mask_blur,
                       imgs=None, skip_nsfw=skip_nsfw,
                       landmarker_score=landmarker_score,
                       enable_swapper=enable_swapper,
                       enable_face_restore=enable_face_restore)
        result = batched_pil_to_tensor(p.init_images)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "FaceFusion": FaceFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceFusion": "FaceFusion",
}
