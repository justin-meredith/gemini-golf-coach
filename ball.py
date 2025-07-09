
import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time
import google.generativeai as genai
import base64
import io
from PIL import Image
import os

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Open the video file
video_path = 'behind-view-slomo.mov'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for Gemini API"""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def analyze_pose_with_gemini(frame, pose_landmarks):
    """Send frame and pose data to Gemini for analysis"""
    try:
        # Convert frame to base64
        frame_b64 = frame_to_base64(frame)
        
        # Create pose description from landmarks
        pose_description = ""
        if pose_landmarks:
            # Key golf swing landmarks
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            
            pose_description = f"""
            Pose Analysis:
            - Head position: ({nose.x:.3f}, {nose.y:.3f})
            - Left shoulder: ({left_shoulder.x:.3f}, {left_shoulder.y:.3f})
            - Right shoulder: ({right_shoulder.x:.3f}, {right_shoulder.y:.3f})
            - Left elbow: ({left_elbow.x:.3f}, {left_elbow.y:.3f})
            - Right elbow: ({right_elbow.x:.3f}, {right_elbow.y:.3f})
            - Left wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f})
            - Right wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f})
            - Left hip: ({left_hip.x:.3f}, {left_hip.y:.3f})
            - Right hip: ({right_hip.x:.3f}, {right_hip.y:.3f})
            """
        
        prompt = f"""
        You are a professional golf instructor analyzing a golf swing. Based on this frame and pose data, provide specific, actionable coaching feedback.
        
        {pose_description}
        
        Focus on:
        1. Posture and alignment
        2. Arm positioning and extension
        3. Hip rotation and weight transfer
        4. Head position and stability
        5. Overall swing mechanics
        
        Provide feedback in 1-2 concise sentences that a golfer can immediately apply. Be specific and constructive.
        """
        
        # Create image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": frame_b64
        }
        
        response = model.generate_content([prompt, image_part])
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def wrap_text(text, font, scale, thickness, max_width):
    """Helper function to wrap feedback text"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]

        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines

# Processing variables
frame_count = 0
process_every_n_frames = int(fps / 1)  # Process at 1 FPS for Gemini (as recommended)
processed_frames = []
current_feedback = ""
feedback_duration = 4 * fps  # Show feedback for 4 seconds
feedback_end_frame = 0

print("Processing video with Gemini AI analysis...")
print("Note: Processing at 1 FPS for optimal Gemini performance")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Process pose detection every frame for smooth visualization
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Send to Gemini for analysis at reduced rate
    if frame_count % process_every_n_frames == 0:
        print(f"Analyzing frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
        gemini_feedback = analyze_pose_with_gemini(frame, results.pose_landmarks)
        if gemini_feedback:
            current_feedback = gemini_feedback
            feedback_end_frame = frame_count + feedback_duration
            print(f"Gemini feedback: {current_feedback}")
    
    # Display current feedback if still active
    if frame_count <= feedback_end_frame and current_feedback:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thickness = 3
        border_thickness = 6
        spacing = 50
        color = (255, 255, 255)
        border = (0, 0, 0)

        max_width = int(width * 0.85)
        wrapped_lines = wrap_text(current_feedback, font, scale, thickness, max_width)
        total_height = len(wrapped_lines) * spacing
        start_y = height - 80 - total_height

        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, font, scale, thickness)[0]
            x = (width - text_size[0]) // 2
            y = start_y + (i * spacing)

            cv2.putText(frame, line, (x, y), font, scale, border, border_thickness, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    
    # Add frame counter for reference
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    processed_frames.append(frame.copy())

cap.release()
cv2.destroyAllWindows()

# Save final video
print("Creating final video with AI analysis...")
output_path = 'gemini_golf_analysis.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in processed_frames:
    out.write(frame)

out.release()
print(f"AI analysis complete! Final video saved to {output_path}")
