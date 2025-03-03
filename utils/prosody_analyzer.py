import os
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import (
    call,
)  # Praat is a gold standard for speech analysis, and parselmouth brings its capabilities to Python
from pydub import AudioSegment
import webrtcvad # Python library for Voice Activity Detection (VAD) # used to detect speech vs. non-speech segments in an audio signal.
from models.prosodic_features import ProsodicFeatures


class ProsodyAnalyzer:
    AUDIO_FOLDER_PATH = "./MIT/interviewee_only_audio/"

    def __init__(self, participant_number):
        self.audio_path = os.path.join(
            self.AUDIO_FOLDER_PATH, f"trimmed_P{participant_number}.wav"
        )
        self.audio_waveform, self.sr = librosa.load(
            self.audio_path, sr=None
        )  # y is the audio waveform as a NumPy array, sr: The sample rate of the audio (e.g., 44100 Hz) # The array elements are the amplitude values at different points in time.
        self.sound = parselmouth.Sound(self.audio_path)
        self.duration = librosa.get_duration(
            y=self.audio_waveform, sr=self.sr
        )  # total duration of the audio in seconds

    def extract_pitch_features(self):
        f0 = librosa.yin(
            self.audio_waveform,
            fmin=50,
            fmax=500,  # fmin and fmax are the minimum and maximum frequencies of humans to be considered for pitch estimation.
        )  # Fundamental frequency (F0) estimation using the YIN algorithm. # f0 is an array of pitch values (in Hz) for each frame of the audio.

        f0 = f0[f0 > 0]  # we keep only the voiced frames (where pitch is meaningful).
        return {  #  If no voiced frames are found (e.g., complete silence), the method returns 0 for all features to avoid errors.
            "F0_MEAN": np.mean(f0) if len(f0) > 0 else 0,
            "F0_MIN": np.min(f0) if len(f0) > 0 else 0,
            "F0_MAX": np.max(f0) if len(f0) > 0 else 0,
            "F0_RANGE": (  #  Shows the variability in pitch. A larger range may indicate expressive speech.
                np.ptp(f0) if len(f0) > 0 else 0
            ),  # difference between max and min pitch
            "F0_SD": (
                np.std(f0) if len(f0) > 0 else 0
            ),  # Measures the consistency of pitch. A high standard deviation indicates frequent pitch changes.
        }

    def extract_intensity_features(self):
        ShortTimeFourierTransform = np.abs(librosa.stft(self.audio_waveform))
        intensity = librosa.amplitude_to_db(
            ShortTimeFourierTransform, ref=np.max
        )  # Converts the amplitude values of the STFT into decibels (dB), which is a logarithmic scale for measuring loudness.
        return {
            "Intensity_MEAN": np.mean(
                intensity
            ),  # Useful for understanding the overall volume.
            "Intensity_MIN": np.min(intensity),
            "Intensity_MAX": np.max(intensity),
            "Intensity_RANGE": np.ptp(
                intensity
            ),  # Shows the dynamic range of the audio. A larger range indicates more variation in loudness.
            "Intensity_SD": np.std(
                intensity
            ),  # Measures the consistency of loudness. A high standard deviation indicates frequent changes in volume.
        }

    def extract_formant_features(self, time_step=0.01):
        formants = self.sound.to_formant_burg(
            time_step=time_step
        )  # computes the formants of the audio signal using Praat’s Burg algorithm.
        f1, f2, f3 = [], [], []
        for time in np.arange(
            0, self.duration, time_step
        ):  # Generates a sequence of time points from 0 to self.duration (total audio length) with a step size of time_step.
            try:
                f1.append(call(formants, "Get value at time", 1, time, "Hertz"))
                f2.append(call(formants, "Get value at time", 2, time, "Hertz"))
                f3.append(call(formants, "Get value at time", 3, time, "Hertz"))
            except:
                continue
        f2_f1_ratio = np.array(f2) / np.array(f1) if len(f1) > 0 else [0]
        f3_f1_ratio = np.array(f3) / np.array(f1) if len(f1) > 0 else [0]
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
            "F2/F1_MEAN": np.mean(f2_f1_ratio),
            "F3/F1_MEAN": np.mean(f3_f1_ratio),
            "F2/F1_SD": np.std(f2_f1_ratio),
            "F3/F1_SD": np.std(f3_f1_ratio),
        }

    def extract_perturbation_features(self):
        pitch = call(self.sound, "To Pitch", 0.0, 50, 500)

        point_process = call(self.sound, "To PointProcess (periodic, cc)...", 50, 500)
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

    def extract_pause_features(self, frame_duration=30):
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
                    if current_pause > 0:
                        pauses.append(current_pause)
                        current_pause = 0

        percent_unvoiced = (len(pauses) / len(frames)) * 100 if len(frames) > 0 else 0

        self.save_audio(silence_audio)

        return {
            "%_Unvoiced": percent_unvoiced,
            "%_Breaks": (len(pauses) / (len(pauses) + 1e-10)) * 100,
            "Max_Pause_Duration": max(pauses) if pauses else 0,
            "Avg_Pause_Duration": np.mean(pauses) if pauses else 0,
        }

    def save_audio(self, audio):
        # Save the silent frames audio
        audio.export(
            "/Users/bassel27/personal_projects/facial_expressions_detection/silence_only.wav",
            format="wav",
        )

    def extract_all_features(self):
        features = {}
        features.update(self.extract_pitch_features())
        features.update(self.extract_intensity_features())
        features.update(self.extract_formant_features())
        features.update(self.extract_perturbation_features())
        features.update(self.extract_pause_features())
        features["Duration"] = self.duration

        return ProsodicFeatures(
        # Pitch Features
        f0_mean=features["F0_MEAN"],
        f0_min=features["F0_MIN"],
        f0_max=features["F0_MAX"],
        f0_range=features["F0_RANGE"],
        f0_sd=features["F0_SD"],

        # Intensity Features
        intensity_mean=features["Intensity_MEAN"],
        intensity_min=features["Intensity_MIN"],
        intensity_max=features["Intensity_MAX"],
        intensity_range=features["Intensity_RANGE"],
        intensity_sd=features["Intensity_SD"],

        # Formant Features
        f1_mean=features["F1_MEAN"],
        f1_sd=features["F1_SD"],
        f2_mean=features["F2_MEAN"],
        f2_sd=features["F2_SD"],
        f3_mean=features["F3_MEAN"],
        f3_sd=features["F3_SD"],
        f2_f1_mean=features["F2/F1_MEAN"],
        f3_f1_mean=features["F3/F1_MEAN"],
        f2_f1_sd=features["F2/F1_SD"],
        f3_f1_sd=features["F3/F1_SD"],

        # Perturbation Features
        jitter=features["Jitter"],
        shimmer=features["Shimmer"],

        # Pause Features
        percent_unvoiced=features["%_Unvoiced"],
        percent_breaks=features["%_Breaks"],
        max_pause_duration=features["Max_Pause_Duration"],
        avg_pause_duration=features["Avg_Pause_Duration"],

        # Duration
        duration=features["Duration"],
    )
