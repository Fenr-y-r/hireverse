from dataclasses import dataclass


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