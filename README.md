# FaceFuison extension for StableDiffusion Webui

[FaceFusion](https://github.com/facefusion/facefusion) is a very nice face swapper and enhancer.

This repo makes it an extension of [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) or [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) Webui.

![Preview](.github/preview.png)

## Installation

-   Clone this repo into `stable-diffusion-webui/extensions` folder.
-   Start the Webui.

## SD webui alwayson_scripts api

```python
alwayson_scripts = {
	"facefusion": {
		"args": [
			# face image
			get_base64_image('./t1.png'),
			# Enable
			True,
			# Execution Provider
			"cuda",
			# Face Detector Score
			0.65,
			# Face Mask Blur
			0.7,
			# Batch Sources Images
			[get_base64_image('./t2.png'), get_base64_image('./t3.png')],
			# Skip check nsfw
			True,
			# Landmarker Score
			0.5
		]
	}
}
```
