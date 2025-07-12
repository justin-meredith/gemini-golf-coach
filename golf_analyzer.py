
#!/usr/bin/env python3
"""
Golf Swing Analysis Pipeline
Processes video to create clean pose visualization and detailed swing report
"""

import sys
import os
from video_processor import VideoProcessor
from report_generator import ReportGenerator

def main():
    # Configuration
    video_path = 'behind-view-full-speed.mov'  # Default video
    
    # Allow command line video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        print("Usage: python golf_analyzer.py [video_path]")
        return
    
    print("🏌️ Golf Swing Analysis Pipeline Starting...")
    print(f"📹 Processing video: {video_path}")
    
    # Step 1: Process video and extract key frames
    print("\n1️⃣ Processing video and detecting swing phases...")
    processor = VideoProcessor(video_path)
    frames_data = processor.process_video()
    
    if not frames_data:
        print("❌ No swing phases detected. Please check video quality and ensure golfer is visible.")
        return
    
    print(f"✅ Detected {len(frames_data)} key swing positions")
    
    # Step 2: Generate analysis report
    print("\n2️⃣ Generating analysis report with AI feedback...")
    
    # Check for API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not found in environment!")
        print("Please set your Gemini API key in Secrets.")
        return
    
    report_generator = ReportGenerator()
    report_path = report_generator.generate_report(frames_data, os.path.basename(video_path))
    
    print("\n🎉 Analysis Complete!")
    print("\n📄 Generated Files:")
    print("  • output/pose_only_video.mp4 - Clean pose landmarks video")
    print(f"  • {report_path} - Detailed swing analysis report")
    print("  • key_frames/ - Individual frame captures")
    
    print("\n💡 Next Steps:")
    print("  1. Review the markdown report for detailed feedback")
    print("  2. Watch the pose-only video to see your swing mechanics")
    print("  3. Use the feedback to practice specific improvements")

if __name__ == "__main__":
    main()
