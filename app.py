from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import uuid
from werkzeug.utils import secure_filename
from pathlib import Path
import threading
import queue
from pipeline import DogBehaviorPipeline
import traceback
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global pipeline instance
pipeline = None
analysis_queue = queue.Queue()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_pipeline():
    global pipeline
    try:
        print("Initializing Dog Behavior Pipeline...")
        pipeline = DogBehaviorPipeline()
        print("Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        traceback.print_exc()
        return False

def analyze_video_async(video_path, task_id, sid):
    """Run video analysis in background thread with real-time updates"""
    try:
        print(f"Starting analysis for task {task_id}")
        
        # Emit initial status
        socketio.emit('status', {'message': 'Starting analysis...', 'progress': 0}, room=sid)
        
        # Process video frame by frame for real-time feedback
        result = pipeline.predict_behavior_realtime(video_path, task_id, socketio, sid)
        
        # Store result
        analysis_results[task_id] = result
        print(f"Analysis completed for task {task_id}")
        
        # Emit completion
        socketio.emit('complete', result, room=sid)
        
    except Exception as e:
        print(f"Analysis failed for task {task_id}: {e}")
        analysis_results[task_id] = {
            "status": "error",
            "message": str(e)
        }
        socketio.emit('error', {'message': str(e)}, room=sid)
    finally:
        # Clean up uploaded file
        try:
            os.remove(video_path)
        except:
            pass

# In-memory storage for results (use database in production)
analysis_results = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400

    try:
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save file
        file.save(filepath)

        # Check file size
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({'error': 'File too large. Maximum size: 500MB'}), 400

        # Start analysis in background
        task_id = unique_id
        analysis_results[task_id] = {"status": "processing"}

        # Get socket ID from request (passed via query param or header)
        sid = request.args.get('sid') or request.headers.get('X-Socket-ID')
        
        thread = threading.Thread(target=analyze_video_async, args=(filepath, task_id, sid))
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': 'Video uploaded and analysis started',
            'status': 'processing'
        })

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def get_analysis_status(task_id):
    if task_id not in analysis_results:
        return jsonify({'error': 'Task not found'}), 404

    result = analysis_results[task_id]
    return jsonify(result)

@app.route('/api/analyze/<task_id>', methods=['GET'])
def get_analysis_result(task_id):
    """Alias for status endpoint for backward compatibility"""
    return get_analysis_status(task_id)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_analysis')
def handle_start_analysis(data):
    video_path = data.get('video_path')
    if not video_path or not os.path.exists(video_path):
        emit('error', {'message': 'Video file not found'})
        return
    
    task_id = str(uuid.uuid4())
    analysis_results[task_id] = {"status": "processing"}
    
    thread = threading.Thread(target=analyze_video_async, args=(video_path, task_id, request.sid))
    thread.daemon = True
    thread.start()
    
    emit('analysis_started', {'task_id': task_id})

if __name__ == "__main__":
    import os
    import eventlet
    import eventlet.wsgi

    port = int(os.environ.get("PORT", 5000))
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", port)), app)