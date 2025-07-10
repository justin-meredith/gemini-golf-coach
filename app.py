
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import json
import subprocess
import threading
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
from video_manager import VideoManager

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'golf_coach_secret_key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output', exist_ok=True)

# Global variables for processing status
processing_status = {}
vm = VideoManager()

ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    coaching_mode = request.form.get('mode', 'hybrid')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create processing job
        job_id = f"job_{timestamp}"
        processing_status[job_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'Video uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'mode': coaching_mode,
            'output_path': None
        }
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'filename': filename,
            'mode': coaching_mode
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process/<job_id>')
def process_video(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_status[job_id]
    if job['status'] != 'uploaded':
        return jsonify({'error': 'Job already processed or in progress'}), 400
    
    # Start processing in background
    thread = threading.Thread(target=process_video_background, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})

def process_video_background(job_id):
    job = processing_status[job_id]
    job['status'] = 'processing'
    job['progress'] = 10
    job['message'] = 'Initializing analysis...'
    
    try:
        filepath = job['filepath']
        mode = job['mode']
        
        print(f"Processing job {job_id}: file={filepath}, mode={mode}")
        
        # Check if uploaded file exists
        if not os.path.exists(filepath):
            job['status'] = 'error'
            job['message'] = f'Uploaded file not found: {filepath}'
            print(f"ERROR: File not found: {filepath}")
            return
        
        # Update video path in the appropriate script
        if mode == 'hybrid':
            script_path = 'hybrid_golf_coach.py'
            output_name = f'hybrid_coaching_analysis_{job_id}.mp4'
        elif mode == 'detailed':
            script_path = 'ball.py'
            output_name = f'gemini_golf_analysis_{job_id}.mp4'
        else:  # realtime mode - not applicable for uploaded videos
            job['status'] = 'error'
            job['message'] = 'Real-time mode requires live camera feed'
            return
        
        # Check if script exists
        if not os.path.exists(script_path):
            job['status'] = 'error'
            job['message'] = f'Analysis script not found: {script_path}'
            print(f"ERROR: Script not found: {script_path}")
            return
        
        # Create a modified version of the script for this specific video
        temp_script = f"temp_{job_id}.py"
        print(f"Creating temporary script: {temp_script}")
        modify_script_for_video(script_path, temp_script, filepath, output_name)
        
        job['progress'] = 20
        job['message'] = 'Starting AI analysis...'
        
        # Run the analysis
        print(f"Running analysis with: python {temp_script}")
        result = subprocess.run(['python', temp_script], 
                              capture_output=True, text=True, timeout=600)
        
        print(f"Analysis completed. Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
        print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
        
        # Clean up temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        if result.returncode == 0:
            output_path = f'output/{output_name}'
            print(f"Checking for output file: {output_path}")
            if os.path.exists(output_path):
                job['status'] = 'completed'
                job['progress'] = 100
                job['message'] = 'Analysis completed successfully!'
                job['output_path'] = output_path
                print(f"SUCCESS: Output saved to {output_path}")
            else:
                job['status'] = 'error'
                job['message'] = 'Analysis completed but output file not found'
                print(f"ERROR: Expected output file {output_path} not found")
                # List files in output directory for debugging
                output_files = os.listdir('output') if os.path.exists('output') else []
                print(f"Files in output directory: {output_files}")
        else:
            job['status'] = 'error'
            job['message'] = f'Analysis failed: {result.stderr[:200]}'
            print(f"ERROR: Analysis failed with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        job['status'] = 'error'
        job['message'] = 'Analysis timed out (10 minutes)'
        print(f"ERROR: Analysis timed out for job {job_id}")
    except Exception as e:
        job['status'] = 'error'
        job['message'] = f'Processing error: {str(e)[:200]}'
        print(f"ERROR: Exception in processing: {e}")
        import traceback
        traceback.print_exc()

def modify_script_for_video(source_script, temp_script, video_path, output_name):
    """Modify the script to use the uploaded video"""
    with open(source_script, 'r') as f:
        content = f.read()
    
    # Add environment check at the top
    env_check = """
# Check for required environment variables
import os
if not os.environ.get('GEMINI_API_KEY'):
    print("ERROR: GEMINI_API_KEY not found in environment!")
    print("Please set your Gemini API key in Secrets.")
    exit(1)

"""
    
    # Replace video path
    if 'ball.py' in source_script:
        content = content.replace("video_path = 'videos/behind-view-full-speed-4.mov'", 
                                f"video_path = '{video_path}'")
        content = content.replace("output_path = 'output/gemini_golf_analysis.mp4'",
                                f"output_path = 'output/{output_name}'")
    else:  # hybrid_golf_coach.py
        content = content.replace("video_path = 'videos/behind-view-full-speed-4.mov'",
                                f"video_path = '{video_path}'")
        content = content.replace("output_path = 'output/hybrid_coaching_analysis.mp4'",
                                f"output_path = 'output/{output_name}'")
    
    # Add environment check after imports
    import_end = content.find('\n\n# Configure Gemini API')
    if import_end == -1:
        import_end = content.find('\n\n# Initialize MediaPipe')
    
    if import_end != -1:
        content = content[:import_end] + '\n' + env_check + content[import_end:]
    
    with open(temp_script, 'w') as f:
        f.write(content)

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_status[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'output_ready': job['status'] == 'completed' and job['output_path'] is not None
    })

@app.route('/download/<job_id>')
def download_result(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_status[job_id]
    if job['status'] != 'completed' or not job['output_path']:
        return jsonify({'error': 'Result not ready'}), 400
    
    if os.path.exists(job['output_path']):
        return send_file(job['output_path'], as_attachment=True, 
                        download_name=f"golf_analysis_{job_id}.mp4")
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/realtime')
def realtime_coach():
    return render_template('realtime.html')

@app.route('/start_realtime')
def start_realtime():
    """Start real-time coaching (this would typically run the camera script)"""
    return jsonify({
        'message': 'Real-time coaching would start here',
        'note': 'This requires camera access and runs the realtime_golf_coach.py script'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
