import os
import librosa
import numpy as np
import parselmouth
from librosa.effects import trim
from parselmouth.praat import (
    call,
)  # Praat is a gold standard for speech analysis, and parselmouth brings its capabilities to Python
from pydub import AudioSegment
import sys
from utils.utils import BASE_DIR
import webrtcvad  # Python library for Voice Activity Detection (VAD) # used to detect speech vs. non-speech segments in an audio signal.
from schemas.model_features import ProsodicFeatures
from pathlib import Path


class ProsodyAnalyzer:
    AUDIO_FOLDER_PATH = AUDIO_FILE_PATH = BASE_DIR + f"/data/raw/audio"

    def __init__(self, participant_id: str):
        self.audio_path = os.path.join(
            self.AUDIO_FOLDER_PATH, f"trimmed_{participant_id}.wav"
        )
        self.audio_waveform, self.sr = librosa.load(
            self.audio_path, sr=16000, mono=True, res_type="kaiser_best"
        )  # audio waveform is a NumPy array, sr: The sample rate of the audio (e.g., 44100 Hz) # The array elements are the amplitude values at different points in time.
        self.audio_waveform = librosa.util.normalize(self.audio_waveform)
        self.sound = parselmouth.Sound(self.audio_waveform, sampling_frequency=self.sr)
        self.duration = librosa.get_duration(
            y=self.audio_waveform, sr=self.sr
        )  # total duration of the audio in seconds

    def _extract_pitch_features(self):
        pitch = self.sound.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]  # Remove unvoiced frames

        if len(f0) == 0:
            return {k: 0 for k in ["F0_MEAN", "F0_MIN", "F0_MAX", "F0_RANGE", "F0_SD"]}

        return {
            "F0_MEAN": np.mean(f0),
            "F0_MIN": np.min(f0),
            "F0_MAX": np.max(f0),
            "F0_RANGE": np.ptp(f0),
            "F0_SD": np.std(f0),
        }

    def _extract_intensity_features(self):
        # Calculate frame-wise RMS energy (time-domain intensity)
        frame_length = 2048  # 93ms at 22050Hz (common for speech)
        hop_length = 512
        rms_energy = librosa.feature.rms(
            y=self.audio_waveform, frame_length=frame_length, hop_length=hop_length
        )[
            0
        ]  # Get 1D array

        # Convert to dB for perceptual loudness (optional but matches your original approach)
        rms_db = librosa.amplitude_to_db(rms_energy, ref=np.max)  # ref=1.0 for dBFS

        return {
            "Intensity_MEAN": np.mean(rms_db),
            "Intensity_MIN": np.min(rms_db),
            "Intensity_MAX": np.max(rms_db),
            "Intensity_RANGE": np.ptp(rms_db),
            "Intensity_SD": np.std(rms_db),
        }

    def _extract_formant_features(self, time_step=0.01):
        F1_DEFAULT = 400  # Schwa vowel average
        F2_DEFAULT = 1500
        F3_DEFAULT = 2500
        formants = self.sound.to_formant_burg(
            time_step=time_step
        )  # computes the formants of the audio signal using Praat’s Burg algorithm.
        f1, f2, f3 = [], [], []

        for time in np.arange(0, self.duration, time_step):
            try:
                # Use parselmouth's built-in NaN handling
                f1_val = formants.get_value_at_time(1, time)
                f2_val = formants.get_value_at_time(2, time)
                f3_val = formants.get_value_at_time(3, time)
                if not np.isnan(f1_val):  # Filter out NaN values
                    f1.append(f1_val)
                    f2.append(f2_val)
                    f3.append(f3_val)
            except:
                continue

        # Handle empty arrays
        f1 = np.array(f1) if f1 else np.array([F1_DEFAULT])
        f2 = np.array(f2) if f2 else np.array([F2_DEFAULT])
        f3 = np.array(f3) if f3 else np.array([F3_DEFAULT])

        # Safe division
        with np.errstate(divide="ignore", invalid="ignore"):
            f2_f1_ratio = np.divide(f2, f1, out=np.zeros_like(f2), where=f1 != 0)
            f3_f1_ratio = np.divide(f3, f1, out=np.zeros_like(f3), where=f1 != 0)
        return {
            "F1_MEAN": np.mean(f1),
            "F1_SD": np.std(
                f1
            ),  #  High F1_SD = Large changes in tongue height (e.g., dynamic speech) # Low F1_SD = Stable tongue height (e.g., monotone speech).
            "F2_MEAN": np.mean(f2),
            "F2_SD": np.std(
                f2
            ),  # High F2_SD = Large changes in tongue backness (e.g., varied vowel sounds). # Low F2_SD = Stable tongue backness (e.g., repetitive speech)
            "F3_MEAN": np.mean(f3),
            "F3_SD": np.std(
                f3
            ),  # High F3_SD = Large changes in lip/pharynx shape (e.g., varied articulation). # Low F3_SD = Stable lip/pharynx shape (e.g., consistent articulation).
            "F2_F1_MEAN": np.mean(f2_f1_ratio),  # f2/f1_mean
            "F3_F1_MEAN": np.mean(f3_f1_ratio),
            "F2_F1_SD": np.std(f2_f1_ratio),
            "F3_F1_SD": np.std(f3_f1_ratio),
        }

    def extract_perturbation_features(self):
        # pitch = call(self.sound, "To Pitch", 0.0, 50, 500)

        point_process = call(self.sound, "To PointProcess (periodic, cc)...", 75, 600)
        return {
            "Jitter": call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            ),  # High jitter indicates voice instability or hoarseness. # It is often used to assess voice quality and detect voice disorders (e.g., vocal fold pathology).
            "Shimmer": call(
                [self.sound, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            ),  # The cycle-to-cycle variability in amplitude.  # High shimmer indicates amplitude instability or voice roughness.
        }

    def _extract_pause_features(self, frame_duration=30):
        vad = webrtcvad.Vad(
            1
        )  # Mode 0: Least aggressive (more likely to classify segments as speech). # Mode 3: Most aggressive (more likely to classify segments as non-speech).
        # Load audio in mono, 16kHz
        audio = (
            AudioSegment.from_wav(self.audio_path).set_frame_rate(16000).set_channels(1)
        )

        # Convert audio to raw samples
        raw_samples = audio.get_array_of_samples()
        frame_size = int(16000 * (frame_duration / 1000))  # Number of samples per frame
        frames = np.array_split(raw_samples, int(len(audio) / frame_duration))

        pauses, current_pause = [], 0
        silence_audio = AudioSegment.silent(
            duration=0, frame_rate=16000
        )  # Empty silent track
        total_speech_segments = 0
        for i, frame in enumerate(frames):
            frame_bytes = frame.astype(np.int16).tobytes()  # Convert to PCM 16-bit
            if len(frame_bytes) == frame_size * 2:  # Ensure correct frame size
                frame_contains_speech = vad.is_speech(frame_bytes, sample_rate=16000)
                if not frame_contains_speech:
                    current_pause += frame_duration / 1000
                    silence_audio += AudioSegment(
                        frame_bytes, sample_width=2, frame_rate=16000, channels=1
                    )
                else:
                    total_speech_segments += 1
                    if current_pause > 0:
                        pauses.append(current_pause)
                        current_pause = 0

        total_pause_duration = sum(pauses)
        percent_unvoiced = (total_pause_duration / self.duration) * 100

        return {
            "percent_Unvoiced": percent_unvoiced,
            "percent_Breaks": (len(pauses) / (total_speech_segments + 1e-10))
            * 100,  # how frequently someone stops speaking (breaks in fluency).
            "max_Pause_Duration": max(pauses) if pauses else 0,
            "avg_Pause_Duration": np.mean(pauses) if pauses else 0,
        }

    def _save_audio(self, audio):
        # Save the silent frames audio
        audio.export(
            "./silence_only.wav",
            format="wav",
        )

    def extract_all_features(self):
        """
        Return: a dictionary of all prosodic features extracted from the audio.
        """
        features = {}
        features.update(self._extract_pitch_features())
        features.update(self._extract_intensity_features())

        features.update(self._extract_formant_features())
        features.update(self.extract_perturbation_features())
        features.update(self._extract_pause_features())
        features["Duration"] = self.duration
        features = {k.lower(): v for k, v in features.items()}
        return ProsodicFeatures(**features)
