
"""
Configuration settings for the golf swing analysis system
"""

# Gemini API Settings
GEMINI_MODEL = "gemini-1.5-flash"
ANALYSIS_INTERVAL = 30  # frames between analysis (30 = ~1 second at 30fps)

# Video Processing Settings
PROCESS_FPS = 1  # FPS for video analysis (Gemini recommended: 1 FPS)
FEEDBACK_DURATION = 4  # seconds to show each feedback
VIDEO_QUALITY = 85  # JPEG quality for Gemini (70-85 recommended)

# Display Settings
FONT_SCALE = 1.5
FONT_THICKNESS = 3
BORDER_THICKNESS = 6
TEXT_COLOR = (255, 255, 255)  # White
BORDER_COLOR = (0, 0, 0)     # Black

# Pose Detection Settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Real-time Settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ANALYSIS_QUEUE_SIZE = 2

# Golf-specific prompts
DETAILED_ANALYSIS_PROMPT = """
You are a PGA golf instructor analyzing a golf swing. Based on this frame and pose data, provide specific, actionable coaching feedback.

Focus on:
1. Posture and alignment
2. Arm positioning and extension  
3. Hip rotation and weight transfer
4. Head position and stability
5. Overall swing mechanics

Provide feedback in 1-2 concise sentences that a golfer can immediately apply. Be specific and constructive.
"""

REALTIME_ANALYSIS_PROMPT = """
You are a PGA golf instructor providing real-time swing coaching. Analyze this golf swing position and provide ONE specific, actionable tip.

Focus on the most important improvement needed right now. Keep response to 1 sentence, maximum 15 words.
Be encouraging but specific.
"""

# Swing phases for advanced analysis
SWING_PHASES = {
    "setup": "Address position and pre-swing setup",
    "takeaway": "Initial movement away from ball",
    "backswing": "Full backswing to top position", 
    "transition": "Change of direction from backswing to downswing",
    "downswing": "Downward motion toward impact",
    "impact": "Club-ball contact zone",
    "follow_through": "Post-impact continuation"
}
