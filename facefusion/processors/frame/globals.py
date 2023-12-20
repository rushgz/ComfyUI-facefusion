from typing import List, Optional

from facefusion.processors.frame.typings import FaceSwapperModel, FaceEnhancerModel, FrameEnhancerModel

face_swapper_model : Optional[FaceSwapperModel] = None
face_enhancer_model : Optional[FaceEnhancerModel] = None
face_enhancer_blend : Optional[int] = None
frame_enhancer_model : Optional[FrameEnhancerModel] = None
frame_enhancer_blend : Optional[int] = None
