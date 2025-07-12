
import google.generativeai as genai
import base64
import io
from PIL import Image
import cv2
import os
import time

class ReportGenerator:
    def __init__(self):
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def frame_to_base64(self, frame_path):
        """Convert image file to base64 for Gemini API"""
        with open(frame_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def analyze_swing_phase(self, frame_path, phase_name, pose_landmarks):
        """Analyze swing phase with Gemini"""
        try:
            frame_b64 = self.frame_to_base64(frame_path)
            
            # Create pose description
            pose_description = ""
            if pose_landmarks:
                landmarks = pose_landmarks.landmark
                left_shoulder = landmarks[11]  # LEFT_SHOULDER
                right_shoulder = landmarks[12]  # RIGHT_SHOULDER
                left_wrist = landmarks[15]  # LEFT_WRIST
                right_wrist = landmarks[16]  # RIGHT_WRIST
                left_hip = landmarks[23]  # LEFT_HIP
                right_hip = landmarks[24]  # RIGHT_HIP
                
                pose_description = f"""
                Pose Data for {phase_name}:
                - Shoulder alignment: Left ({left_shoulder.x:.3f}, {left_shoulder.y:.3f}), Right ({right_shoulder.x:.3f}, {right_shoulder.y:.3f})
                - Wrist positions: Left ({left_wrist.x:.3f}, {left_wrist.y:.3f}), Right ({right_wrist.x:.3f}, {right_wrist.y:.3f})
                - Hip position: Left ({left_hip.x:.3f}, {left_hip.y:.3f}), Right ({right_hip.x:.3f}, {right_hip.y:.3f})
                """
            
            # Phase-specific prompts
            phase_prompts = {
                "setup": "Analyze the golfer's setup position. Focus on posture, alignment, and pre-swing fundamentals.",
                "takeaway_parallel": "Analyze the takeaway when the club is parallel to the ground. Focus on the initial movement and club path.",
                "top_backswing": "Analyze the top of the backswing position. Focus on shoulder turn, arm position, and overall coil.",
                "downswing_halfway": "Analyze the downswing when club is parallel to ground. Focus on hip rotation and weight transfer.",
                "impact": "Analyze the impact position. Focus on body position at ball contact and club delivery.",
                "follow_through_halfway": "Analyze the follow-through. Focus on extension and rotation through the ball.",
                "finish": "Analyze the finish position. Focus on balance, completion of rotation, and overall follow-through."
            }
            
            prompt = f"""
            You are a friendly golf instructor analyzing a student's swing. Look at this {phase_name} position and provide encouraging, conversational feedback.

            {pose_description}

            {phase_prompts.get(phase_name, "Analyze this golf swing position.")}

            Provide feedback in a warm, encouraging tone as if speaking directly to the golfer. Keep it conversational and specific - 2-3 sentences maximum. Focus on what they're doing well and one key improvement they could make.

            Example style: "Nice setup position! Your posture looks solid and athletic. Try widening your stance just a bit more for even better stability."
            """
            
            image_part = {
                "mime_type": "image/jpeg",
                "data": frame_b64
            }
            
            response = self.model.generate_content([prompt, image_part])
            return response.text.strip()
            
        except Exception as e:
            print(f"Error analyzing {phase_name}: {e}")
            return f"Could not analyze {phase_name} - please check the image quality."
    
    def generate_report(self, frames_data, video_name):
        """Generate complete markdown report"""
        if not frames_data:
            print("No frame data provided for report generation")
            return
        
        # Sort frames by typical swing sequence
        phase_order = [
            "setup", "takeaway_parallel", "top_backswing", 
            "downswing_halfway", "impact", "follow_through_halfway", "finish"
        ]
        
        sorted_frames = []
        for phase in phase_order:
            for frame_data in frames_data:
                if frame_data['phase'] == phase:
                    sorted_frames.append(frame_data)
                    break
        
        # Generate report content
        report_content = f"""# Golf Swing Analysis Report

## Video: {video_name}

This report analyzes your golf swing at 7 key positions, providing personalized feedback to help improve your technique.

---

"""
        
        for i, frame_data in enumerate(sorted_frames, 1):
            phase = frame_data['phase']
            filename = frame_data['filename']
            frame_number = frame_data['frame_number']
            pose_landmarks = frame_data['pose_landmarks']
            
            print(f"Analyzing {phase}...")
            
            # Get Gemini analysis
            feedback = self.analyze_swing_phase(filename, phase, pose_landmarks)
            
            # Format phase name for display
            display_phase = phase.replace('_', ' ').title()
            
            report_content += f"""## {i}. {display_phase}

![{display_phase}]({filename})

**Frame {frame_number}**

{feedback}

---

"""
            
            # Small delay to be respectful to API
            time.sleep(1)
        
        report_content += """
## Summary

This analysis covers the fundamental positions of your golf swing. Focus on the specific feedback for each phase, and remember that small adjustments can lead to significant improvements. Practice these positions slowly and build muscle memory for consistency.

### Next Steps
- Work on the specific improvements mentioned for each phase
- Practice these positions in slow motion
- Record another swing to track your progress

*Generated with AI-powered golf swing analysis*
"""
        
        # Save report
        report_path = 'output/swing_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to {report_path}")
        return report_path
