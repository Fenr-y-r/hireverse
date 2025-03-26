# Hireverse
<p align="center">
  <img src="https://github.com/user-attachments/assets/112deae5-a9f6-46cd-b127-560955a0517e" width="571" alt="image">
</p>

This is an AI platform that, first, enables users to practice for job interviews and, second, helps companies in their recruitment process by conducting initial interviews with their applicants. The system has an AI avatar that conducts video interviews, asking questions, and also analyzes user performance through  speech, facial expressions, and responses. After the interview, the AI provides feedback about  both behavioral and technical aspects. This includes engagement, emotional cues, facial expressions, tone, confidence, and corrections for both behavioral and technical answers

# Features
## 1. Feature Extraction
Three types of features are extracted for the model training: prosodic, facial and lexical features. These features are extracted from the dataset using multiprocessing.
### A. Facial Features
This module extracts facial features per frame using MediaPipe. It detects face landmarks, aligns faces, extracts regions of interest (ROI), and computes aggregated facial metrics such as:
  - Smile intensity (DeepFace)
  - Face alignment (eye positioning)
  - Facial landmarks & distances (brow, eye, lip)
  - Head pose (face angles via solvePnP)

<img width="400" alt="image" src="https://github.com/user-attachments/assets/9a62e4bc-6baf-4b9d-8e53-dcd60b8278c7"/> <img width="400" alt="image" src="https://github.com/user-attachments/assets/2c8e37ca-9d66-4231-9419-46026e2c6ad4" />

## B. Prosodic Features
This extracts key prosodic features from interview audio to analyze speech expressiveness, fluency, and articulation. It combines Praat, Librosa, and WebRTC VAD for accurate speech analysis in interview assessments. Since these features are extracted per frame, they undergo statistical aggregation (mean, min, max, std) to get per video interview value. Prosodic features include:
  - Pitch
  - Intensity
  - Speaking rate
  - Pause duration & frequency
  - Jitter & shimmer (voice stability)

## C. Lexical Features

# 2. Avatar

# 3. Behavioral Questions Evalution

# 4. Technical Questions Evalution

# 5. Web App
Using Figma, React and Mongo DB

# Dataset Citation
This project utilizes the following dataset:
I. Naim, I. Tanveer, D. Gildea, M. E. Hoque, Automated Analysis and Prediction of Job Interview Performance, to appear in IEEE Transactions on Affective Computing.

<img src="https://github.com/user-attachments/assets/836201c4-81c4-4283-81a5-7ec33408ac0e" width="500">

