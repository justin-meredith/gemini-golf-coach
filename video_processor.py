
import cv2
import mediapipe as mp
import numpy as np
import os
from swing_phase_detector import SwingPhaseDetector

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.phase_detector = SwingPhaseDetector()
        self.frames_data = []
        
    def process_video(self):
        """Process video and extract key frames"""
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if video needs rotation
        needs_rotation = original_width > original_height
        if needs_rotation:
            width = original_height
            height = original_width
        else:
            width = original_width
            height = original_height
        
        print(f"Processing video: {width}x{height} at {original_fps} FPS")
        
        # Create output directories
        os.makedirs("key_frames", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        frame_count = 0
        pose_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Rotate frame if needed
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Process pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Detect swing phase
                phase = self.phase_detector.detect_phase(results.pose_landmarks, frame_count)
                
                # Check if we should capture this frame
                if self.phase_detector.should_capture_frame(phase, frame_count):
                    # Save original frame for report
                    frame_filename = f"key_frames/{phase}_frame_{frame_count}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    
                    # Store frame data
                    self.frames_data.append({
                        'phase': phase,
                        'frame_number': frame_count,
                        'filename': frame_filename,
                        'pose_landmarks': results.pose_landmarks
                    })
                    
                    print(f"Captured {phase} at frame {frame_count}")
                
                # Create clean pose frame for video
                pose_frame = np.zeros_like(frame)
                self.mp_drawing.draw_landmarks(
                    pose_frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
                )
                pose_frames.append(pose_frame)
            else:
                # Add black frame if no pose detected
                pose_frames.append(np.zeros_like(frame))
        
        cap.release()
        
        # Create clean pose video
        self.create_pose_video(pose_frames, original_fps, width, height)
        
        return self.frames_data
    
    def create_pose_video(self, pose_frames, fps, width, height):
        """Create clean video with only pose landmarks"""
        output_path = 'output/pose_only_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in pose_frames:
            out.write(frame)
        
        out.release()
        print(f"Clean pose video saved to {output_path}")
    
    def get_processed_frames(self):
        """Return processed frame data"""
        return self.frames_data
