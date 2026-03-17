# Dog Behavior Analysis Web App

A web application for analyzing dog behavior from video using AI models. This app provides both a REST API and a web interface for uploading videos and getting behavior analysis results.

## Features

- **Video Upload**: Support for MP4, AVI, MOV, and MKV formats (up to 500MB)
- **AI Analysis**: Uses YOLOv8 for pose detection and BiLSTM for behavior classification
- **Real-time Progress**: Live status updates during analysis
- **Web Interface**: Clean, responsive frontend for easy video upload and result viewing
- **REST API**: Programmatic access to analysis functionality

## Setup

### Prerequisites

- Python 3.8+
- Required model files:
  - `best_large.pt` (YOLOv8 pose model)
  - `smoothed_bilstm_cnn_hybrid.keras` (BiLSTM behavior classifier)
  - `scaler.pkl` (Feature scaler)
  - `label_encoder.pkl` (Label encoder)

### Installation

1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone or download this repository

3. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   uv sync
   ```

5. Ensure all model files are in the same directory as the scripts

### Running the Application

Start the Flask server:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

### POST /api/upload
Upload a video file for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `video` field with video file

**Response:**
```json
{
  "task_id": "unique-task-identifier",
  "message": "Video uploaded and analysis started",
  "status": "processing"
}
```

### GET /api/status/{task_id}
Get the analysis status and results.

**Response (processing):**
```json
{
  "status": "processing"
}
```

**Response (completed):**
```json
{
  "status": "success",
  "class": "behavior_name",
  "class_id": 0,
  "confidence": 0.95,
  "probabilities": {
    "behavior_1": 0.95,
    "behavior_2": 0.03,
    "behavior_3": 0.02
  }
}
```

**Response (error):**
```json
{
  "status": "error",
  "message": "Error description"
}
```

## Usage

### Web Interface

1. Open `http://localhost:5000` in your browser
2. Click or drag a video file to the upload area
3. Click "Analyze Video" to start processing
4. View real-time progress and final results

### API Usage

```python
import requests

# Upload video
with open('dog_video.mp4', 'rb') as f:
    response = requests.post('http://localhost:5000/api/upload',
                           files={'video': f})
    task_id = response.json()['task_id']

# Check status
while True:
    response = requests.get(f'http://localhost:5000/api/status/{task_id}')
    result = response.json()
    if result['status'] != 'processing':
        break
    time.sleep(2)

# Print results
if result['status'] == 'success':
    print(f"Behavior: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## Architecture

- **Backend**: Flask web server with REST API
- **AI Pipeline**: YOLOv8 pose detection + Kalman filtering + BiLSTM classification
- **Frontend**: Vanilla HTML/CSS/JavaScript with responsive design
- **Processing**: Asynchronous video analysis with background threads

## Model Details

- **Pose Detection**: YOLOv8-pose for keypoint extraction
- **Smoothing**: Kalman filtering for keypoint trajectory smoothing
- **Classification**: BiLSTM-CNN hybrid model for behavior recognition
- **Features**: Position, velocity, acceleration, and relative distances

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: The app forces CPU usage for TensorFlow. If you have GPU memory issues, ensure your GPU has sufficient VRAM.

2. **Model files not found**: Ensure all required model files are in the same directory as the scripts.

3. **Large video files**: Videos are processed in memory. For very large files, consider reducing resolution or implementing streaming processing.

4. **Port already in use**: Change the port in `app.py` if 5000 is occupied.

### Performance Tips

- Use videos with 30 FPS or less for faster processing
- Shorter videos (10-30 seconds) give better results
- Ensure good lighting and clear dog visibility in videos

## License

This project is for educational and research purposes.</content>
<parameter name="filePath">/mnt/Cheta HD/CHETA-WORK/Codes/Dogsense GUI/README.md