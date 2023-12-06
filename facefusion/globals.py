from typing import List, Optional

from facefusion.typing import FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, FaceDetectorModel, FaceRecognizerModel, Padding

# general
source_path : Optional[str] = None
target_path : Optional[str] = None
output_path : Optional[str] = None
# misc
skip_download : Optional[bool] = None
# execution
execution_providers : List[str] = []
execution_thread_count : Optional[int] = None
execution_queue_count : Optional[int] = None
max_memory : Optional[int] = None
# face analyser
face_analyser_order : Optional[FaceAnalyserOrder] = None
face_analyser_age : Optional[FaceAnalyserAge] = None
face_analyser_gender : Optional[FaceAnalyserGender] = None
face_detector_model : Optional[FaceDetectorModel] = None
face_detector_size : Optional[str] = None
face_detector_score : Optional[float] = None
face_recognizer_model : Optional[FaceRecognizerModel] = None
# face selector
face_selector_mode : Optional[FaceSelectorMode] = None
reference_face_position : Optional[int] = None
reference_face_distance : Optional[float] = None
reference_frame_number : Optional[int] = None
# face mask
face_mask_blur : Optional[float] = None
face_mask_padding : Optional[Padding] = None
# output creation
output_image_quality : Optional[int] = None
# frame processors
frame_processors : List[str] = []
