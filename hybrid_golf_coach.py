
import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import base64
import io
from PIL import Image
import os
import time
import threading
import queue
import pandas as pd
from video_manager import VideoManager

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

class HybridGolfCoach:
    def __init__(self, user_id="demo_user"):
        self.current_feedback = ""
        self.feedback_timestamp = 0
        self.feedback_duration = 5  # seconds
        self.analysis_queue = queue.Queue(maxsize=2)
        self.running = True
        self.user_id = user_id
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Initialize video manager for cloud storage
        self.video_manager = VideoManager()
        
        # Load script bank
        self.script_bank = self.load_script_bank()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_worker)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def load_script_bank(self):
        """Load the coaching script bank from CSV"""
        try:
            df = pd.read_csv('attached_assets/Refined_Golf_Swing_Feedback_Script_Bank__10th_Grade_Advanced_Tone__1752085164319.csv')
            return df.to_dict('records')
        except Exception as e:
            print(f"Could not load script bank: {e}")
            return []
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64 string for Gemini API"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=70)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def format_script_bank_context(self):
        """Format script bank examples for the prompt"""
        if not self.script_bank:
            return ""
        
        examples = []
        for i, script in enumerate(self.script_bank[:5]):  # Use first 5 examples
            examples.append(f"""
Example {i+1}: {script.get('Fault', 'Unknown fault')}
- Observation: "{script.get('Observation', '')}"
- Beginner Cue: "{script.get('Cue (Beginner)', '')}"
- Advanced Cue: "{script.get('Cue (Advanced)', '')}"
- Reinforcement: "{script.get('Reinforcement (Beginner)', '')}"
            """)
        
        return "\n".join(examples)
    
    def analyze_with_gemini(self, frame, pose_landmarks):
        """Analyze pose with Gemini API using hybrid approach"""
        try:
            frame_b64 = self.frame_to_base64(frame)
            
            pose_description = ""
            if pose_landmarks:
                landmarks = pose_landmarks.landmark
                
                # Extract key landmarks for golf analysis
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                
                # Calculate key metrics
                shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
                
                pose_description = f"""
                Pose Analysis:
                - Head position: ({nose.x:.3f}, {nose.y:.3f})
                - Shoulder alignment: {shoulder_slope:.3f}
                - Left arm: Elbow ({left_elbow.x:.3f}, {left_elbow.y:.3f}), Wrist ({left_wrist.x:.3f}, {left_wrist.y:.3f})
                - Right arm: Elbow ({right_elbow.x:.3f}, {right_elbow.y:.3f}), Wrist ({right_wrist.x:.3f}, {right_wrist.y:.3f})
                - Hip position: Left ({left_hip.x:.3f}, {left_hip.y:.3f}), Right ({right_hip.x:.3f}, {right_hip.y:.3f})
                """
            
            script_context = self.format_script_bank_context()
            
            prompt = f"""
            You are a professional golf instructor and swing coach providing real-time coaching feedback. 
            
            Use the following coaching framework as your foundation, but adapt dynamically to what you observe:
            
            COACHING FRAMEWORK:
            1. OBSERVATION: Describe what you see objectively
            2. CUE/FIX: Give ONE specific, actionable instruction
            3. REINFORCEMENT: End with encouraging context about why this helps
            
            COACHING STYLE EXAMPLES:
            {script_context}
            
            CURRENT SWING ANALYSIS:
            {pose_description}
            
            Your response should follow this structure:
            1. Identify the most important issue you observe
            2. Provide a clear, feel-based cue (avoid technical jargon)
            3. End with positive reinforcement
            
            Keep your response conversational and encouraging. Focus on ONE key improvement.
            Adapt the language to what seems appropriate for this golfer's skill level.
            Maximum 2 sentences total.
            """
            
            image_part = {
                "mime_type": "image/jpeg",
                "data": frame_b64
            }
            
            response = model.generate_content([prompt, image_part])
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return None
    
    def analysis_worker(self):
        """Background thread for Gemini analysis"""
        while self.running:
            try:
                if not self.analysis_queue.empty():
                    frame, pose_landmarks, timestamp = self.analysis_queue.get(timeout=1)
                    
                    feedback = self.analyze_with_gemini(frame, pose_landmarks)
                    if feedback:
                        self.current_feedback = feedback
                        self.feedback_timestamp = timestamp
                        print(f"Hybrid feedback: {feedback}")
                
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis worker error: {e}")
    
    def add_frame_for_analysis(self, frame, pose_landmarks):
        """Add frame to analysis queue (non-blocking)"""
        try:
            if self.analysis_queue.qsize() < 2:
                self.analysis_queue.put_nowait((frame.copy(), pose_landmarks, time.time()))
        except queue.Full:
            pass
    
    def get_current_feedback(self):
        """Get current feedback if still valid"""
        if time.time() - self.feedback_timestamp < self.feedback_duration:
            return self.current_feedback
        return ""
    
    def stop(self):
        """Stop the analysis thread"""
        self.running = False

def main():
    # Check if we should process a video file (for web app) or use camera
    video_path = 'videos/behind-view-full-speed-4.mov'  # Default video for testing
    use_camera = not os.path.exists(video_path) or len(os.sys.argv) > 1
    
    if use_camera:
        # Initialize camera for real-time coaching
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        process_realtime_coaching(cap)
    else:
        # Process video file for web app
        process_video_file(video_path)

def process_realtime_coaching(cap):
    """Handle real-time camera coaching"""
    
    coach = HybridGolfCoach()
    frame_count = 0
    analysis_interval = 30  # Analyze every 30 frames
    
    print("Hybrid Golf Coach started!")
    print("Using script bank foundation with dynamic analysis")
    print("Press 'q' to quit, 'space' to trigger analysis")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Send for analysis periodically
                if frame_count % analysis_interval == 0:
                    coach.add_frame_for_analysis(frame, results.pose_landmarks)
            
            # Display current feedback
            feedback = coach.get_current_feedback()
            if feedback:
                # Add background for better readability
                cv2.rectangle(frame, (10, 10), (630, 80), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (630, 80), (0, 255, 0), 2)
                
                # Wrap text if needed
                words = feedback.split()
                if len(' '.join(words)) > 50:
                    mid = len(words) // 2
                    line1 = ' '.join(words[:mid])
                    line2 = ' '.join(words[mid:])
                    cv2.putText(frame, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, line2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, feedback, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Hybrid Golf Coach - Script Bank + Dynamic Analysis", 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'space' for instant analysis", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Hybrid Golf Coach', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and results.pose_landmarks:
                coach.add_frame_for_analysis(frame, results.pose_landmarks)
    
    finally:
        coach.stop()
        cap.release()
        cv2.destroyAllWindows()

def process_video_file(video_path):
    """Process uploaded video file for web app"""
    print(f"Processing video file: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    output_path = 'output/hybrid_coaching_analysis.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Process at reduced FPS for analysis
    slowdown_factor = 2
    fps = max(1, original_fps // slowdown_factor)
    
    # Check for rotation
    needs_rotation = original_width > original_height
    if needs_rotation:
        width, height = original_height, original_width
    else:
        width, height = original_width, original_height
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    coach = HybridGolfCoach()
    frame_count = 0
    process_every_n_frames = max(1, fps // 1)  # Process every second
    current_feedback = ""
    feedback_end_frame = 0
    feedback_duration = 4 * fps
    
    print(f"Processing {total_frames} frames at {fps} FPS...")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Rotate if needed
            if needs_rotation:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Process pose detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Send for analysis periodically
                if frame_count % process_every_n_frames == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Analyzing frame {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    feedback = coach.analyze_with_gemini(frame, results.pose_landmarks)
                    if feedback and "failed" not in feedback.lower():
                        current_feedback = feedback
                        feedback_end_frame = frame_count + feedback_duration
                        print(f"Hybrid feedback: {feedback}")
            
            # Display current feedback
            if frame_count <= feedback_end_frame and current_feedback:
                # Add text overlay similar to ball.py
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.2
                thickness = 2
                border_thickness = 4
                color = (255, 255, 255)
                border = (0, 0, 0)
                
                # Wrap long feedback text
                max_width = int(width * 0.85)
                words = current_feedback.split()
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
                
                # Draw text lines
                spacing = 40
                total_height = len(lines) * spacing
                start_y = height - 60 - total_height
                
                for i, line in enumerate(lines):
                    text_size = cv2.getTextSize(line, font, scale, thickness)[0]
                    x = (width - text_size[0]) // 2
                    y = start_y + (i * spacing)
                    
                    cv2.putText(frame, line, (x, y), font, scale, border, border_thickness, cv2.LINE_AA)
                    cv2.putText(frame, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            
            # Add frame counter
            cv2.putText(frame, f"Hybrid Coaching - Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        coach.stop()
        
        print(f"Hybrid coaching analysis complete! Output saved to {output_path}")
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        cap.release()
        out.release()
        coach.stop()

if __name__ == "__main__":
    main()
