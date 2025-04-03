import os
import sys
import numpy as np
from pydub import AudioSegment
import assemblyai as aai
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lexical_features import LexicalFeatures
from LIWC import *
import re
import csv
from collections import Counter
import spacy
import string

# Define filler words and sentence start words
filler_words = {"uhm", "um", "uh"}  # Count these anywhere
sentence_start_words = {"basically", "like"}  # Only count these at sentence starts

# AssemblyAI transcription configuration
config = aai.TranscriptionConfig(
    disfluencies=True,
    speaker_labels=True,
    speakers_expected=2
)

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

class LexicalAnalyser:
    def __init__(self, audio_path: str):
        """
        Initialize the lexicalanalyser with a custom audio file path.
        """
        self.audio_path = audio_path
        self.audio = AudioSegment.from_file(self.audio_path)
        self.duration = self.audio.duration_seconds

    def _extract_lexical_features(self):
        """
        Extract lexical features from the audio file.
        """
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(self.audio_path, config=config)
        words = transcript.split()
        total_words = len(words)
        unique_words = len(set(words))
        filler_count = sum(1 for word in words if word in filler_words)
        print(f"Filler words: {filler_count}")
        return {
            "Total_words": total_words,
            "Unique_words": unique_words,
            "Filler_words": filler_count,
        }

    def _extract_LIWC_features(self):
        """
        Extract LIWC features from the audio file.
        """
        liwc_features = {
            "Individual": sum(1 for word in words if word in Individual_Words),
            "Group": sum(1 for word in words if word in Group_Words),
            "They": sum(1 for word in words if word in They_Words),
            "PosEmotion": sum(1 for word in words if word in PosEmotion),
            "NegEmotion": sum(1 for word in words if word in NegEmotion),
            "Anxiety": sum(1 for word in words if word in Anxiety),
            "Anger": sum(1 for word in words if word in Anger),
            "Sadness": sum(1 for word in words if word in Sadness),
            "Cognitive": sum(1 for word in words if word in Cognitive),
            "Inhibition": sum(1 for word in words if word in Inhibition),
            "Preceptual": sum(1 for word in words if word in Preceptual),
            "Relativity": sum(1 for word in words if word in Relativity),
            "Work": sum(1 for word in words if word in Work),
            "Swear": sum(1 for word in words if word in Swear),
            "Articles": sum(1 for word in words if word in Articles),
            "Verbs": sum(1 for token in doc if token.pos_ == "VERB"),
            "Adverbs": sum(1 for token in doc if token.pos_ == "ADV"),
            "Adpositions": sum(1 for token in doc if token.pos_ == "ADP"),
            "Conjunctions": sum(1 for token in doc if token.pos_ in ["CCONJ", "SCONJ"]),
            "Negations": sum(1 for word in words if word in Negations),
            "Quantifiers": sum(1 for word in words if word in Quantifiers),
            "Numbers": sum(1 for token in doc if token.pos_ == "NUM")
                      + sum(1 for token in doc if re.match(r'\b\d+(st|nd|rd|th)\b', token.text))
                      + sum(1 for token in doc if "%" in token.text),
        }
        return liwc_features

    def extract_all_features(self):
        """
        Extract all features (lexical and LIWC) from the audio file.
        """
        features = {}
        features.update(self._extract_lexical_features())
        features.update(self._extract_LIWC_features())
        features["Duration"] = self.duration
        features = {k.lower(): v for k, v in features.items()}
        return LexicalFeatures(**features)
