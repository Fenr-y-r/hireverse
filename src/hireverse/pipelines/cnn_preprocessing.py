import random
import uuid
import cv2
from hireverse.utils.utils import *
from hireverse.utils.face_analyzer import FaceAnalyzer
from hireverse.schemas.frame import Frame
from typing import List


import concurrent.futures
import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import numpy as np

def get_processed_frames( vid_file_path: str):

    face_analyzer = FaceAnalyzer()
    frames = face_analyzer.get_video_frames(participant_id="randomId",video_path= vid_file_path, target_fps=20)
    filtered_frames: List[Frame] = []
    for frame in frames:
        frame.facial_landmarks_obj  = face_analyzer.process_image_results(frame.image)
        if frame.facial_landmarks_obj:
            frame.facial_landmarks = frame.facial_landmarks_obj.landmark
            filtered_frames.append(frame)
    frames = filtered_frames 

    for frame in frames:
        new_width = 640
        _, width = frame.image.shape[:2]
        if width != new_width:
            frame.resize(new_width=new_width)

        frame.align_face_with_mediapipe_landmarks()

        frame.face = face_analyzer.get_face_coordinates(frame.facial_landmarks, frame.image)
        x, y, w, h = frame.face
        frame.crop_frame(x, y, x + w, y+h)

        frame.image =  cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    
        frame.image = np.clip(frame.image.astype('float32') / 255.0, 0.0, 1.0)
    
        frame.resize(new_width=640, new_height=640)

    for frame in frames[:7]:
        frame.reset_drawable_image()
        # frame.draw_face_border()
        # frame.draw_facial_landmarks()
        # if frame.facial_landmarks:
        #     frame.draw_circle_at_facial_landmark(frame.facial_landmarks[10], frame.facial_landmarks[152])
        frame.display()
    
    return frames


# def process_single_video(vid_path, output_dir):
#     """Process one video and save results to disk"""
#     try:
#         frames = get_processed_frames(vid_path)
        
#         # Convert frames to numpy arrays
#         frame_data = np.stack([frame.image for frame in frames])
        
#         # Save with video metadata
#         vid_name = Path(vid_path).stem
#         output_path = os.path.join(output_dir, f"{vid_name}.npz")
        
#         np.savez_compressed(
#             output_path,
#             frames=frame_data,
#             video_id=vid_name,
#             num_frames=len(frames)
        
#         return True
#     except Exception as e:
#         print(f"Error processing {vid_path}: {str(e)}")
#         return False

# def process_all_videos(video_paths, output_dir, max_workers=8):
#     """Process videos in parallel with progress tracking"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Use ProcessPoolExecutor for CPU-bound tasks
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(process_single_video, path, output_dir): path
#             for path in video_paths
#         }
        
#         # Track progress
#         results = []
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_paths)):
#             results.append(future.result())
    
#     print(f"Successfully processed {sum(results)}/{len(video_paths)} videos")

# # Example Usage
# if __name__ == "__main__":
#     video_dir = "path/to/your/videos"
#     output_dir = "processed_frames"
    
#     video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
#                   if f.endswith(('.mp4', '.avi', '.mov'))]
    
#     process_all_videos(video_paths[:139], output_dir)  # Process first 139 videos