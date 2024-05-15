

class PostprocessedImage:
    def __init__(self, image):
        self.image = image
        self.info = {}
        self.extra_images = []
        self.nametags = []
        self.disable_processing = False
        self.caption = None
