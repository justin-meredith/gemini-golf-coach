
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
        enhanced_frames = []
        club_path_points = []  # Store club head positions for trail
        
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
                
                # Create enhanced frame with pose overlay and club path
                enhanced_frame = frame.copy()
                
                # Draw pose landmarks on actual video
                self.mp_drawing.draw_landmarks(
                    enhanced_frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2)
                )
                
                # Estimate club head position and add to trail
                club_head_pos = self.estimate_club_head_position(results.pose_landmarks, width, height)
                if club_head_pos:
                    club_path_points.append(club_head_pos)
                    
                    # Keep only recent trail points (last 30 frames for smooth trail)
                    if len(club_path_points) > 30:
                        club_path_points.pop(0)
                    
                    # Draw club path trail
                    self.draw_club_path(enhanced_frame, club_path_points)
                
                enhanced_frames.append(enhanced_frame)
            else:
                # Add original frame if no pose detected
                enhanced_frames.append(frame.copy())
        
        cap.release()
        
        # Create enhanced video with pose overlay and club path
        self.create_enhanced_video(enhanced_frames, original_fps, width, height)
        
        return self.frames_data
    
    def create_enhanced_video(self, enhanced_frames, fps, width, height):
        """Create enhanced video with pose overlay and club path tracking"""
        output_path = 'output/enhanced_swing_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in enhanced_frames:
            out.write(frame)
        
        out.release()
        print(f"Enhanced swing video saved to {output_path}")
    
    def estimate_club_head_position(self, pose_landmarks, width, height):
        """Estimate club head position based on hand positions and arm extension"""
        landmarks = pose_landmarks.landmark
        
        # Get hand positions (club grip)
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Average hand position (grip)
        grip_x = (left_wrist.x + right_wrist.x) / 2
        grip_y = (left_wrist.y + right_wrist.y) / 2
        
        # Estimate club length and direction
        # Calculate angle from shoulder to grip
        shoulder_to_grip_x = grip_x - left_shoulder.x
        shoulder_to_grip_y = grip_y - left_shoulder.y
        
        # Extend from grip position to estimate club head (assuming 3 feet club extension)
        club_length_ratio = 0.15  # Approximate club length relative to body
        
        # Direction vector from shoulder through grip
        direction_x = shoulder_to_grip_x
        direction_y = shoulder_to_grip_y
        
        # Normalize and extend
        length = np.sqrt(direction_x**2 + direction_y**2)
        if length > 0:
            direction_x /= length
            direction_y /= length
            
            # Estimate club head position
            club_head_x = grip_x + direction_x * club_length_ratio
            club_head_y = grip_y + direction_y * club_length_ratio
            
            # Convert to pixel coordinates
            pixel_x = int(club_head_x * width)
            pixel_y = int(club_head_y * height)
            
            # Ensure within frame bounds
            pixel_x = max(0, min(width-1, pixel_x))
            pixel_y = max(0, min(height-1, pixel_y))
            
            return (pixel_x, pixel_y)
        
        return None
    
    def draw_club_path(self, frame, club_path_points):
        """Draw club path trail on the frame"""
        if len(club_path_points) < 2:
            return
        
        # Draw trail with fading effect
        for i in range(1, len(club_path_points)):
            # Calculate opacity based on point age (newer points more opaque)
            opacity = i / len(club_path_points)
            
            # Draw line segment
            thickness = max(1, int(opacity * 4))
            color_intensity = int(255 * opacity)
            
            cv2.line(frame, 
                    club_path_points[i-1], 
                    club_path_points[i], 
                    (0, color_intensity, 255),  # Orange to red trail
                    thickness)
        
        # Draw current club head position as a bright circle
        if club_path_points:
            current_pos = club_path_points[-1]
            cv2.circle(frame, current_pos, 6, (0, 255, 255), -1)  # Bright yellow circle
            cv2.circle(frame, current_pos, 8, (0, 0, 255), 2)    # Red border
    
    def get_processed_frames(self):
        """Return processed frame data"""
        return self.frames_data
