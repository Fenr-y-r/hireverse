from dataclasses import dataclass


@dataclass
class ProsodicFeatures:
    # Pitch Features
    f0_mean: float
    f0_min: float
    f0_max: float
    f0_range: float
    f0_sd: float

    # Intensity Features
    intensity_mean: float
    intensity_min: float
    intensity_max: float
    intensity_range: float
    intensity_sd: float

    # Formant Features
    f1_mean: float
    f1_sd: float
    f2_mean: float
    f2_sd: float
    f3_mean: float
    f3_sd: float
    f2_f1_mean: float
    f3_f1_mean: float
    f2_f1_sd: float
    f3_f1_sd: float

    # Perturbation Features
    jitter: float
    shimmer: float

    # Pause Features
    percent_unvoiced: float
    percent_breaks: float
    max_pause_duration: float
    avg_pause_duration: float

    # Duration
    duration: float

    def __str__(self):
        return (
            f"ProsodicFeatures("
            f"f0_mean={self.f0_mean}, f0_min={self.f0_min}, f0_max={self.f0_max}, f0_range={self.f0_range}, f0_sd={self.f0_sd}, "
            f"intensity_mean={self.intensity_mean}, intensity_min={self.intensity_min}, intensity_max={self.intensity_max}, intensity_range={self.intensity_range}, intensity_sd={self.intensity_sd}, "
            f"f1_mean={self.f1_mean}, f1_sd={self.f1_sd}, f2_mean={self.f2_mean}, f2_sd={self.f2_sd}, f3_mean={self.f3_mean}, f3_sd={self.f3_sd}, "
            f"f2_f1_mean={self.f2_f1_mean}, f3_f1_mean={self.f3_f1_mean}, f2_f1_sd={self.f2_f1_sd}, f3_f1_sd={self.f3_f1_sd}, "
            f"jitter={self.jitter}, shimmer={self.shimmer}, "
            f"percent_unvoiced={self.percent_unvoiced}, percent_breaks={self.percent_breaks}, max_pause_duration={self.max_pause_duration}, avg_pause_duration={self.avg_pause_duration}, "
            f"duration={self.duration})"
        )


@dataclass
class FacialFeatures:
    average_outer_brow_height_mean: float
    average_inner_brow_height_mean: float
    eye_open_mean: float
    outer_lip_height_mean: float
    inner_lip_height_mean: float
    lip_corner_distance_mean: float
    smile_mean: bool
    pitch_mean: float
    yaw_mean: float
    roll_mean: float

    average_outer_brow_height_std: float
    average_inner_brow_height_std: float
    eye_open_std: float
    outer_lip_height_std: float
    inner_lip_height_std: float
    lip_corner_distance_std: float
    smile_std: float
    pitch_std: float
    yaw_std: float
    roll_std: float

    average_outer_brow_height_min: float
    average_inner_brow_height_min: float
    eye_open_min: float
    outer_lip_height_min: float
    inner_lip_height_min: float
    lip_corner_distance_min: float
    smile_min: bool
    pitch_min: float
    yaw_min: float
    roll_min: float

    average_outer_brow_height_max: float
    average_inner_brow_height_max: float
    eye_open_max: float
    outer_lip_height_max: float
    inner_lip_height_max: float
    lip_corner_distance_max: float
    smile_max: bool
    pitch_max: float
    yaw_max: float
    roll_max: float

    average_outer_brow_height_median: float
    average_inner_brow_height_median: float
    eye_open_median: float
    outer_lip_height_median: float
    inner_lip_height_median: float
    lip_corner_distance_median: float
    smile_median: bool
    pitch_median: float
    yaw_median: float
    roll_median: float
    
@dataclass
class LexicalFeatures:
    # Word counts
    Total_words:int
    Unique_words:int
    Filler_words:int

    # Speaking rate
    Total_words_rate:float
    Unique_words_rate:float
    Filler_words_rate:float

    # LIWC
    Individual:int
    We:int
    They:int
    Non_Fluences:int	
    PosEmotion:int	
    NegEmotion:int	
    Anxiety:int
    Anger:int
    Sadness:int	
    Cognitive:int	
    Inhibition:int	
    Preceptual:int	
    Relativity:int
    Work:int
    Swear:int	
    Articles:int
    Verbs:int	
    Adverbs:int
    Prepositions:int
    Conjunctions:int
    Negations:int
    Quantifiers:int
    Numbers:int

    def _str_(self):
        return (
            f"LexicalFeatures("
            f"Total_words={self.Total_words}, Unique_words={self.Unique_words}, Filler_words={self.Filler_words}, "
            f"Total_words_rate={self.Total_words_rate}, Unique_words_rate={self.Unique_words_rate}, Filler_words_rate={self.Filler_words_rate}, "
            f"Individual={self.Individual}, We={self.We}, They={self.They}, Non_Fluences={self.Non_Fluences}, PosEmotion={self.PosEmotion}, "
            f"NegEmotion={self.NegEmotion}, Anxiety={self.Anxiety}, Anger={self.Anger}, Sadness={self.Sadness}, Cognitive={self.Cognitive}, "
            f"Inhibition={self.Inhibition}, Preceptual={self.Preceptual}, Relativity={self.Relativity}, Work={self.Work}, Swear={self.Swear}, "
            f"Articles={self.Articles}, Verbs={self.Verbs}, Adverbs={self.Adverbs}, Prepositions={self.Prepositions}, Conjunctions={self.Conjunctions}, "
            f"Negations={self.Negations}, Quantifiers={self.Quantifiers}, Numbers={self.Numbers})"
        )