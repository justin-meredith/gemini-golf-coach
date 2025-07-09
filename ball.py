import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time

# Load swing data from JSON
with open('swing_data.json', 'r') as f:
    swing_data = json.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video file
video_path = 'behind-view-slomo.mov'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Convert timestamp to frame number
def parse_timestamp(timestamp):
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    seconds = parse_timestamp(timestamp)
    return int(seconds * fps)

# Helper function to wrap feedback text
def wrap_text(text, font, scale, thickness, max_width):
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

# Convert timestamps to frame numbers and add feedback end time
for swing in swing_data['swings']:
    swing['frame_number'] = timestamp_to_frame(swing['timestamp'], fps)
    swing['feedback_end_frame'] = swing['frame_number'] + (4 * fps)  # 4 seconds of feedback

frame_count = 0
process_every_n_frames = int(fps / 20)  # ~3 FPS

processed_frames = []
print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process every nth frame
    if frame_count % process_every_n_frames == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        # (Pose results not used yet — ready for future features)

    # Display feedback if applicable
    current_feedback = None
    for swing in swing_data['swings']:
        if swing['frame_number'] <= frame_count <= swing['feedback_end_frame']:
            current_feedback = swing['feedback']
            break

    # Display feedback overlay
    if current_feedback:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.8
        thickness = 4
        border_thickness = 8
        spacing = 60
        color = (255, 255, 255)
        border = (0, 0, 0)

        max_width = int(width * 0.8)
        wrapped_lines = wrap_text(current_feedback, font, scale, thickness, max_width)
        total_height = len(wrapped_lines) * spacing
        start_y = height - 90 - total_height

        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, font, scale, thickness)[0]
            x = (width - text_size[0]) // 2
            y = start_y + (i * spacing)

            cv2.putText(frame, line, (x, y), font, scale, border, border_thickness, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # Store processed frame
    processed_frames.append(frame.copy())

    # Display live window (optional)
   # cv2.imshow('Golf Swing Analysis', frame)
  #  if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()

# Save final video
print("Creating final video...")
output_path = 'final_golf_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in processed_frames:
    out.write(frame)

out.release()
print(f"Processing complete. Final video saved to {output_path}")
