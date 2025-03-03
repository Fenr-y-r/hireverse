from dataclasses import dataclass, field
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from typing import Optional

@dataclass
class SelectedFacialLandmarks:
    inner_brow_left: NormalizedLandmark
    outer_brow_left: NormalizedLandmark
    inner_brow_right: NormalizedLandmark
    outer_brow_right: NormalizedLandmark
    eye_outer_left: NormalizedLandmark
    eye_outer_right: NormalizedLandmark
    eye_inner_left: NormalizedLandmark
    eye_inner_right: NormalizedLandmark
    outer_lip_height: float
    inner_lip_height: float
    lip_corner_distance: float
    outer_lip_above: Optional[NormalizedLandmark] = None
    outer_lip_below: Optional[NormalizedLandmark] = None
    inner_lip_above: Optional[NormalizedLandmark] = None
    inner_lip_below: Optional[NormalizedLandmark] = None
    lip_corner_right: Optional[NormalizedLandmark] = None
    lip_corner_left: Optional[NormalizedLandmark] = None