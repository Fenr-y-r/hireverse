

class SelectedFacialLandmarks:
    def __init__(
        self,
        inner_brow_left,
        outer_brow_left,
        inner_brow_right,
        outer_brow_right,
        eye_outer_left,
        eye_outer_right,
        eye_inner_left,
        eye_inner_right,
        outer_lip_height,
        inner_lip_height,
        lip_corner_distance,
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
