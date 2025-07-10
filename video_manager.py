
import os
import time
from datetime import datetime
from replit.object_storage import Client

class VideoManager:
    def __init__(self):
        """Initialize video manager with Object Storage client"""
        self.client = Client()
        self.video_folders = {
            'uploads': 'user_uploads/',      # Original swing videos
            'analyzed': 'analyzed_videos/',  # AI-processed videos
            'thumbnails': 'thumbnails/',     # Video preview images
            'sessions': 'sessions/'          # Training session data
        }
        
        # Create folder structure
        self._ensure_folders_exist()
    
    def _ensure_folders_exist(self):
        """Ensure all necessary folders exist in Object Storage"""
        for folder_type, folder_path in self.video_folders.items():
            try:
                # Try to list objects in folder to check if it exists
                list(self.client.list(prefix=folder_path, limit=1))
            except:
                # If folder doesn't exist, create a marker file
                self.client.upload_from_text(f"{folder_path}.keep", "folder marker")
    
    def upload_user_video(self, video_file_path, user_id, session_id=None):
        """Upload user's swing video to Object Storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_part = f"_{session_id}" if session_id else ""
        
        # Create organized path: user_uploads/user123/20240109_143022_session1.mov
        storage_path = f"{self.video_folders['uploads']}{user_id}/{timestamp}{session_part}.mov"
        
        try:
            with open(video_file_path, 'rb') as video_file:
                self.client.upload_from_bytes(storage_path, video_file.read())
            
            print(f"✓ Uploaded: {storage_path}")
            return storage_path
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return None
    
    def save_analyzed_video(self, analyzed_video_path, original_storage_path):
        """Save AI-analyzed video to Object Storage"""
        # Mirror the path structure but in analyzed folder
        analyzed_path = original_storage_path.replace(
            self.video_folders['uploads'], 
            self.video_folders['analyzed']
        ).replace('.mov', '_analyzed.mp4')
        
        try:
            with open(analyzed_video_path, 'rb') as video_file:
                self.client.upload_from_bytes(analyzed_path, video_file.read())
            
            print(f"✓ Analyzed video saved: {analyzed_path}")
            return analyzed_path
        except Exception as e:
            print(f"✗ Analyzed video save failed: {e}")
            return None
    
    def get_video_url(self, storage_path):
        """Get a downloadable URL for a video"""
        try:
            # Generate a signed URL that's valid for 1 hour
            url = self.client.download_url(storage_path, expires_in=3600)
            return url
        except Exception as e:
            print(f"✗ URL generation failed: {e}")
            return None
    
    def download_video(self, storage_path, local_path):
        """Download video from Object Storage to local file"""
        try:
            video_data = self.client.download_as_bytes(storage_path)
            with open(local_path, 'wb') as f:
                f.write(video_data)
            return True
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return False
    
    def list_user_videos(self, user_id, video_type='uploads'):
        """List all videos for a specific user"""
        folder_path = f"{self.video_folders[video_type]}{user_id}/"
        try:
            videos = []
            for obj in self.client.list(prefix=folder_path):
                if not obj.name.endswith('.keep'):
                    videos.append({
                        'name': obj.name,
                        'path': obj.name,
                        'size': obj.size,
                        'modified': obj.updated
                    })
            return videos
        except Exception as e:
            print(f"✗ List videos failed: {e}")
            return []
    
    def delete_video(self, storage_path):
        """Delete video from Object Storage"""
        try:
            self.client.delete(storage_path)
            print(f"✓ Deleted: {storage_path}")
            return True
        except Exception as e:
            print(f"✗ Delete failed: {e}")
            return False
    
    def save_session_data(self, user_id, session_id, session_data):
        """Save training session metadata"""
        session_path = f"{self.video_folders['sessions']}{user_id}/{session_id}.json"
        
        try:
            import json
            session_json = json.dumps(session_data, indent=2)
            self.client.upload_from_text(session_path, session_json)
            print(f"✓ Session data saved: {session_path}")
            return session_path
        except Exception as e:
            print(f"✗ Session save failed: {e}")
            return None

# Usage example for the golf coaching app
def example_usage():
    vm = VideoManager()
    
    # Upload a user's swing video
    storage_path = vm.upload_user_video(
        video_file_path="videos/user_swing.mov",
        user_id="user123",
        session_id="morning_practice"
    )
    
    # Save analyzed video
    analyzed_path = vm.save_analyzed_video(
        analyzed_video_path="output/analyzed_swing.mp4",
        original_storage_path=storage_path
    )
    
    # Get video URLs for web app
    original_url = vm.get_video_url(storage_path)
    analyzed_url = vm.get_video_url(analyzed_path)
    
    # Save session metadata
    session_data = {
        "date": datetime.now().isoformat(),
        "original_video": storage_path,
        "analyzed_video": analyzed_path,
        "feedback_count": 5,
        "improvements": ["posture", "follow_through"]
    }
    
    vm.save_session_data("user123", "morning_practice", session_data)

if __name__ == "__main__":
    example_usage()
