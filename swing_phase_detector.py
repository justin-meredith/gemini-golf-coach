
import numpy as np
import mediapipe as mp

class SwingPhaseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.phase_history = []
        self.key_frames = {}
        
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def get_club_position_angle(self, pose_landmarks):
        """Estimate club position based on arm angles"""
        landmarks = pose_landmarks.landmark
        
        # Use lead arm (left arm for right-handed golfer)
        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Calculate arm angle relative to vertical
        arm_angle = self.calculate_angle(
            type('Point', (), {'x': shoulder.x, 'y': shoulder.y + 0.1})(),  # Point below shoulder
            shoulder,
            wrist
        )
        
        return arm_angle
    
    def detect_phase(self, pose_landmarks, frame_number):
        """Detect current swing phase based on pose landmarks"""
        if not pose_landmarks:
            return "unknown"
        
        landmarks = pose_landmarks.landmark
        
        # Key body points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate metrics
        club_angle = self.get_club_position_angle(pose_landmarks)
        hip_rotation = abs(left_hip.x - right_hip.x)
        shoulder_rotation = abs(left_shoulder.x - right_shoulder.x)
        wrist_height = (left_wrist.y + right_wrist.y) / 2
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        wrist_relative_height = wrist_height - shoulder_height
        
        # Phase detection logic
        phase = "setup"
        
        if club_angle > 140 and wrist_relative_height < -0.1:
            phase = "finish"
        elif club_angle > 120 and wrist_relative_height > 0.1:
            phase = "follow_through_halfway"
        elif club_angle < 45 and abs(wrist_relative_height) < 0.05:
            phase = "impact"
        elif club_angle < 60 and wrist_relative_height > 0:
            phase = "downswing_halfway"
        elif club_angle > 130 and wrist_relative_height < -0.15:
            phase = "top_backswing"
        elif club_angle > 90 and club_angle < 130:
            phase = "takeaway_parallel"
        elif club_angle < 90 and wrist_relative_height > -0.05:
            phase = "setup"
        
        # Store phase history for smoothing
        self.phase_history.append(phase)
        if len(self.phase_history) > 10:
            self.phase_history.pop(0)
        
        # Use most common phase in recent history for stability
        if len(self.phase_history) >= 3:
            phase = max(set(self.phase_history[-3:]), key=self.phase_history[-3:].count)
        
        return phase
    
    def should_capture_frame(self, phase, frame_number):
        """Determine if this frame should be captured as a key frame"""
        target_phases = [
            "setup", "takeaway_parallel", "top_backswing", 
            "downswing_halfway", "impact", "follow_through_halfway", "finish"
        ]
        
        if phase in target_phases and phase not in self.key_frames:
            self.key_frames[phase] = frame_number
            return True
        
        return False
    
    def get_key_frames(self):
        """Return dictionary of captured key frames"""
        return self.key_frames
