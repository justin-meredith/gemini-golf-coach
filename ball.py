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
from video_manager import VideoManager

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
video_path = 'videos/behind-view-full-speed-4.mov'  # Use full speed video
cap = cv2.VideoCapture(video_path)
original_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Slowdown settings
slowdown_factor = 3  # Make video 3x slower (adjust as needed: 2, 3, 4, etc.)
fps = max(1, original_fps // slowdown_factor)  # Ensure minimum 1 FPS

print(f"Original video FPS: {original_fps}")
print(f"Slowed down to: {fps} FPS (factor: {slowdown_factor}x)")

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check if video needs rotation (common with phone videos)
# If width > height but the video appears to be portrait, we need to rotate
needs_rotation = original_width > original_height
if needs_rotation:
    # After rotation, dimensions will be swapped
    width = original_height
    height = original_width
    print(f"Detected rotated video ({original_width}x{original_height}). Will rotate to {width}x{height}")
else:
    width = original_width
    height = original_height
    print(f"Video dimensions: {width}x{height}")

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for Gemini API"""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def analyze_pose_with_gemini(frame, pose_landmarks, max_retries=2):
    """Send frame and pose data to Gemini for analysis with timeout and retries"""
    for attempt in range(max_retries + 1):
        try:
            print(f"  Sending to Gemini API (attempt {attempt + 1}/{max_retries + 1})...")
            start_time = time.time()

            # Check if API key is set
            if not os.environ.get('GEMINI_API_KEY'):
                print("  ERROR: GEMINI_API_KEY not found in environment!")
                return "API key not configured. Please set GEMINI_API_KEY in Secrets."

            # Convert frame to base64
            frame_b64 = frame_to_base64(frame)
            print(f"  Image encoded in {time.time() - start_time:.2f}s")

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
            You are a golf swing analyst providing a detailed technical breakdown to a golf instructor. Analyze this frame and pose data objectively, describing exactly what is happening in the student's swing mechanics.

            {pose_description}

            Provide a factual analysis covering:
            1. Current body position and posture details
            2. Arm angles, extension, and positioning relative to optimal positions
            3. Hip and shoulder rotation angles and weight distribution
            4. Head position and spine angle maintenance
            5. Overall swing phase identification and mechanics assessment

            Be technical and descriptive rather than prescriptive. Report what you observe in 2-3 sentences as if briefing the coach on their student's current swing state.
            """

            # Create image part for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": frame_b64
            }

            # Generate content with timeout simulation
            api_start = time.time()
            response = model.generate_content([prompt, image_part])
            api_time = time.time() - api_start

            print(f"  Gemini API responded in {api_time:.2f}s")
            return response.text.strip()

        except Exception as e:
            print(f"  Gemini API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries + 1} attempts")
                return f"Analysis failed: {str(e)[:50]}..."

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

    # Rotate frame if needed (90 degrees clockwise to fix sideways video)
    if needs_rotation:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Process pose detection every frame for smooth visualization
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Send to Gemini for analysis at reduced rate
    if frame_count % process_every_n_frames == 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = (frame_count / total_frames) * 100
        print(f"\n=== Analyzing frame {frame_count}/{total_frames} ({progress:.1f}%) ===")

        analysis_start = time.time()
        gemini_feedback = analyze_pose_with_gemini(frame, results.pose_landmarks)
        analysis_time = time.time() - analysis_start

        if gemini_feedback and "failed" not in gemini_feedback.lower():
            current_feedback = gemini_feedback
            feedback_end_frame = frame_count + feedback_duration
            print(f"✓ Success in {analysis_time:.1f}s: {current_feedback}")
        else:
            print(f"✗ Failed in {analysis_time:.1f}s: {gemini_feedback or 'No response'}")
            # Continue without feedback for this frame

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
output_path = 'output/gemini_golf_analysis.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in processed_frames:
    out.write(frame)

out.release()
print(f"AI analysis complete! Final video saved to {output_path}")

# Upload to cloud storage for production app
print("Uploading to cloud storage...")
vm = VideoManager()

# Upload original video
original_storage_path = vm.upload_user_video(
    video_file_path=video_path,
    user_id="demo_user",
    session_id=f"analysis_{int(time.time())}"
)

# Upload analyzed video
if original_storage_path:
    analyzed_storage_path = vm.save_analyzed_video(
        analyzed_video_path=output_path,
        original_storage_path=original_storage_path
    )
    
    if analyzed_storage_path:
        print(f"✓ Videos stored in cloud:")
        print(f"  Original: {original_storage_path}")
        print(f"  Analyzed: {analyzed_storage_path}")
        
        # Generate shareable URLs
        original_url = vm.get_video_url(original_storage_path)
        analyzed_url = vm.get_video_url(analyzed_storage_path)
        
        if original_url and analyzed_url:
            print(f"\n📱 Shareable links (valid for 1 hour):")
            print(f"  Original: {original_url}")
            print(f"  Analyzed: {analyzed_url}")
else:
    print("⚠️  Cloud upload failed - check Object Storage setup")