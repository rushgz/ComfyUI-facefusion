import os


class Script:
    pass


def basedir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image
