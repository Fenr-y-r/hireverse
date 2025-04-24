import os
import numpy as np
from pydub import AudioSegment
import assemblyai as aai
import re
import csv
from collections import Counter
import string
import sys
import os
import sys
from pathlib import Path
from src.schemas.model_features import LexicalFeatures
from src.utils.LIWC import *

filler_words = {"uhm", "um", "uh"}  # Count these anywhere
sentence_start_words = {"basically", "like"}  # Only count these at sentence starts
aai.settings.api_key = "52e3e90185574153903f7fb2fb3bb81e"
# AssemblyAI transcription configuration
config = aai.TranscriptionConfig(
    disfluencies=True,
    speaker_labels=True,
    speakers_expected=2
)


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
        self.words = transcript.text.split()
        self.transcript=transcript.text.strip()
        Split=re.findall(r"\b\w+\b", self.transcript.lower())
        total_words = len(self.words)
        unique_words = len(set(self.words))
        filler_count = sum(1 for word in Split if word in filler_words)
        sentence_start_count = 0
        sentences = re.split(r"[.!?]", self.transcript)  # Split by sentence-ending punctuation
        for sentence in sentences:
            first_word_match = re.search(r"\b(\w+)\b", sentence.strip())  # Find first word
            if first_word_match:
                first_word = first_word_match.group(1).lower()
                if first_word in sentence_start_words:
                    sentence_start_count += 1
        self.filler_count = sentence_start_count+filler_count            
        return {
            "Total_words": total_words,
            "Unique_words": unique_words,
            "Filler_words": self.filler_count,
            "Total_words_rate":total_words / self.duration,
            "Unique_words_rate": unique_words / self.duration,
            "Filler_words_rate": self.filler_count / self.duration,
        }
        

    def _extract_LIWC_features(self):
        """
        Extract LIWC features from the audio file.
        """
        words = self.words
        liwc_features = {
            "Individual": sum(1 for word in words if word in Individual_Words),
            "We": sum(1 for word in words if word in Group_Words),
            "They": sum(1 for word in words if word in They_Words),
            "Non_Fluences": self.filler_count,
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
            "Negations": sum(1 for word in words if word in Negations),
            "Quantifiers": sum(1 for word in words if word in Quantifiers),          
        }
        return liwc_features

    def extract_all_features(self):
        """
        Extract all features (lexical and LIWC) from the audio file.
        """
        features = {}
        features.update(self._extract_lexical_features())
        features.update(self._extract_LIWC_features())
        return LexicalFeatures(**features)
