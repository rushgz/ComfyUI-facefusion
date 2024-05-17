# coding=utf-8

import enum
import tempfile
from typing import List, Optional

from facefusion.core import run_new


class DeviceProvider(enum.Enum):
    CPU = "cpu"
    GPU = "cuda"


def swap_face(
    source_paths: [List[str]],
    target_path: str,
    output_path: Optional[str] = None,
    provider: DeviceProvider = DeviceProvider.CPU,
    detector_score: float = 0.65,
    mask_blur: float = 0.7,
    skip_nsfw: bool = True,
    landmarker_score: float = 0.5,
    enable_swapper: bool = True,
    enable_face_restore: bool = True,
) -> Optional[str]:
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    result = run_new(
        source_path=source_paths, target_path=target_path, output_path=output_path,
        provider=provider.value, detector_score=detector_score, mask_blur=mask_blur, skip_nsfw=skip_nsfw,
        landmarker_score=landmarker_score, enable_swapper=enable_swapper, enable_face_restore=enable_face_restore
    )
    return result
