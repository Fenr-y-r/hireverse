from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark



class SelectedFacialLandmarks:
    def __init__(
        self,
        inner_brow_left: NormalizedLandmark,
        outer_brow_left: NormalizedLandmark,
        inner_brow_right: NormalizedLandmark,
        outer_brow_right: NormalizedLandmark,
        eye_outer_left: NormalizedLandmark,
        eye_outer_right: NormalizedLandmark,
        eye_inner_left: NormalizedLandmark,
        eye_inner_right: NormalizedLandmark,
        outer_lip_height: float,
        inner_lip_height: float,
        lip_corner_distance: float,
        **kwargs
    ):
        self.inner_brow_left = inner_brow_left
        self.outer_brow_left = outer_brow_left
        self.inner_brow_right = inner_brow_right
        self.outer_brow_right = outer_brow_right
        self.eye_outer_left = eye_outer_left
        self.eye_outer_right = eye_outer_right
        self.eye_inner_left = eye_inner_left
        self.eye_inner_right = eye_inner_right
        
        self.outer_lip_height = outer_lip_height
        self.inner_lip_height = inner_lip_height
        self.lip_corner_distance = lip_corner_distance

        self.outer_lip_above = kwargs.get("outer_lip_above", None)
        self.outer_lip_below = kwargs.get("outer_lip_below", None)
        self.inner_lip_above = kwargs.get("inner_lip_above", None)
        self.inner_lip_below = kwargs.get("inner_lip_below", None)
        self.lip_corner_right = kwargs.get("lip_corner_right", None)
        self.lip_corner_left = kwargs.get("lip_corner_left", None)

    def __str__(self):
        return (
            f"SelectedFacialLandmarks("
            f"inner_brow_left={self.inner_brow_left}, "
            f"outer_brow_left={self.outer_brow_left}, "
            f"inner_brow_right={self.inner_brow_right}, "
            f"outer_brow_right={self.outer_brow_right}, "
            f"eye_outer_left={self.eye_outer_left}, "
            f"eye_outer_right={self.eye_outer_right}, "
            f"eye_inner_left={self.eye_inner_left}, "
            f"eye_inner_right={self.eye_inner_right}, "
            f"outer_lip_height={self.outer_lip_height}, "
            f"inner_lip_height={self.inner_lip_height}, "
            f"lip_corner_distance={self.lip_corner_distance}, "
            f"outer_lip_above={self.outer_lip_above}, "
            f"outer_lip_below={self.outer_lip_below}, "
            f"inner_lip_above={self.inner_lip_above}, "
            f"inner_lip_below={self.inner_lip_below}, "
            f"lip_corner_right={self.lip_corner_right}, "
            f"lip_corner_left={self.lip_corner_left})"
        )