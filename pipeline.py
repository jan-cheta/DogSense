# Configure GPU memory growth - DISABLE GPU USAGE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU devices
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Reduce logging

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

# Model paths
YOLO_MODEL_PATH = "best_large.pt"
BILSTM_MODEL_PATH = "smoothed_bilstm_cnn_hybrid.keras"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Kalman Filter Parameters
STATE_SIZE = 4
MEASUREMENT_SIZE = 2
DEFAULT_DT = 1.0

# Kalman Filter Configuration
MEASUREMENT_NOISE_STD = 20.0
R = np.array([[MEASUREMENT_NOISE_STD**2, 0], 
              [0, MEASUREMENT_NOISE_STD**2]], dtype=np.float32)

PROCESS_POS_STD = 1.0
PROCESS_VEL_STD = 1.0
Q = np.array([
    [PROCESS_POS_STD**2, 0, 0, 0],
    [0, PROCESS_POS_STD**2, 0, 0],
    [0, 0, PROCESS_VEL_STD**2, 0],
    [0, 0, 0, PROCESS_VEL_STD**2]
], dtype=np.float32)

P0 = np.diag([100.0, 100.0, 25.0, 25.0]).astype(np.float32)
MAHALANOBIS_THRESH = 9.21
MAX_ACCEPTABLE_JUMP = 2000.0
MAX_INNOVATION_NORM = 5000.0

class DogBehaviorPipeline:
    def __init__(self, yolo_path=YOLO_MODEL_PATH, bilstm_path=BILSTM_MODEL_PATH,
                 scaler_path=SCALER_PATH, label_encoder_path=LABEL_ENCODER_PATH):
        """Initialize the pipeline with model paths"""
        self.yolo_model = None
        self.bilstm_model = None
        self.scaler = None
        self.label_encoder = None
        
        try:
            print("Loading YOLO model...")
            if not os.path.exists(yolo_path):
                print(f"Model {yolo_path} not found, downloading...")
                self.yolo_model = YOLO(yolo_path)  # This will download if not exists
            else:
                self.yolo_model = YOLO(yolo_path)
            
            print("Loading BiLSTM model...")
            self.bilstm_model = load_model(bilstm_path)
            
            print("Loading feature scaler...")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("Loading label encoder...")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load models and preprocessors: {str(e)}")
    
    def create_kalman_filter(self, dt=DEFAULT_DT):
        """Create and initialize Kalman filter"""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)

        kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE)
        kf.transitionMatrix = F
        kf.measurementMatrix = H
        kf.processNoiseCov = Q.copy()
        kf.measurementNoiseCov = R.copy()
        kf.errorCovPost = P0.copy()
        return kf
    
    def mahalanobis_distance(self, innov, S):
        """Calculate Mahalanobis distance for outlier detection"""
        try:
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            invS = np.linalg.pinv(S)
        m = float(innov.T.dot(invS).dot(innov))
        return m
    
    def apply_kalman_smoothing(self, keypoints):
        """Apply Kalman filtering to keypoint sequence"""
        if not keypoints or len(keypoints) == 0:
            return []
            
        # Get number of keypoints from first valid frame
        num_kp = next((len(f) for f in keypoints if f is not None and len(f) > 0), 0)
        if num_kp == 0:
            return []
            
        # Create Kalman filters for each keypoint
        kalman_filters = [self.create_kalman_filter() for _ in range(num_kp)]
        
        # Initialize using first valid frame
        first_valid = next((f for f in keypoints if f is not None and len(f)>0), None)
        if first_valid:
            for i, pt in enumerate(first_valid):
                kf = kalman_filters[i]
                kf.statePost = np.array([[pt[0]], [pt[1]], [0.0], [0.0]], dtype=np.float32)
        
        smoothed = []
        
        # Process frames
        for frame in keypoints:
            smoothed_frame = []
            for i, kf in enumerate(kalman_filters):
                state_pre = kf.predict()
                pred_x = float(kf.statePre[0,0])
                pred_y = float(kf.statePre[1,0])
                
                has_meas = (frame is not None and i < len(frame))
                if has_meas:
                    meas_x = float(frame[i][0])
                    meas_y = float(frame[i][1])
                    meas = np.array([[meas_x],[meas_y]], dtype=np.float32)
                    
                    # Innovation
                    H = kf.measurementMatrix
                    x_pre = kf.statePre.reshape(-1,1)
                    z_pred = (H.dot(x_pre)).astype(np.float32)
                    innov = (meas - z_pred).astype(np.float32)
                    
                    # Innovation covariance
                    P_pre = kf.errorCovPre if hasattr(kf, 'errorCovPre') else kf.errorCovPost
                    S = H.dot(P_pre).dot(H.T) + kf.measurementNoiseCov
                    
                    # Mahalanobis gating
                    m = self.mahalanobis_distance(innov, S)
                    if np.isnan(m) or m > MAHALANOBIS_THRESH or np.linalg.norm(innov) > MAX_INNOVATION_NORM:
                        x = pred_x
                        y = pred_y
                    else:
                        kf.correct(meas)
                        x = float(kf.statePost[0,0])
                        y = float(kf.statePost[1,0])
                        if abs(x - meas_x) > MAX_ACCEPTABLE_JUMP or abs(y - meas_y) > MAX_ACCEPTABLE_JUMP:
                            kf.statePost = np.array([[meas_x],[meas_y],[0.0],[0.0]], dtype=np.float32)
                            kf.errorCovPost = P0.copy()
                            x = meas_x
                            y = meas_y
                else:
                    x = pred_x
                    y = pred_y
                    kf.errorCovPost = kf.errorCovPost + np.eye(STATE_SIZE, dtype=np.float32) * 1.0
                
                smoothed_frame.append([float(x), float(y)])
            
            smoothed.append(smoothed_frame)
        
        return smoothed

    def yolo_inference(self, vid_path):
        """Extract keypoints from video using YOLOv8-pose"""
        if not os.path.exists(vid_path):
            raise FileNotFoundError(f"Video file not found: {vid_path}")
            
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {vid_path}")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1 if fps <= 30 else round(fps / 30)
            frames_out = []
            
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_index % frame_interval == 0:
                    # Resize frame for faster inference
                    height, width = frame.shape[:2]
                    if width > 320:
                        scale = 320 / width
                        new_width = 320
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    results = self.yolo_model.predict(frame, conf=0.5, imgsz=320, verbose=False, device='cpu')
                    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                        kps = results[0].keypoints.xy[0].cpu().numpy().tolist()
                        # Scale keypoints back to original frame size if we resized
                        if width > 320:
                            scale_x = width / 320
                            scale_y = height / new_height
                            kps = [[x * scale_x, y * scale_y] for x, y in kps]
                    else:
                        kps = None
                    frames_out.append(kps)
                frame_index += 1
                
            return frames_out
            
        finally:
            cap.release()

    def process_video_features(self, keypoints_list, target_len=32):
        """
        Optimized feature processing for real-time analysis
        """
        # Skip if too few valid frames
        valid = [np.array(f, dtype=np.float32) for f in keypoints_list if f is not None]
        if len(valid) < 5:
            return None
            
        # Stack frames
        kps = np.stack(valid, axis=0)
        
        # Interpolate to target length (pad_or_trim function from feature_extraction.py)
        orig_len = len(kps)
        t_orig = np.linspace(0, 1, orig_len)
        t_new = np.linspace(0, 1, target_len)
        interp = np.empty((target_len, kps.shape[1], kps.shape[2]), dtype=np.float32)
        for j in range(kps.shape[1]):
            interp[:, j, 0] = np.interp(t_new, t_orig, kps[:, j, 0])
            interp[:, j, 1] = np.interp(t_new, t_orig, kps[:, j, 1])
        kps = interp
        
        # Global normalization (from feature_extraction.py)
        min_xy = np.min(kps.reshape(-1, 2), axis=0)
        max_xy = np.max(kps.reshape(-1, 2), axis=0)
        scale = np.maximum(max_xy - min_xy, 1e-6)
        kps_norm = (kps - min_xy) / scale
        
        # Compute velocities and accelerations
        vel = np.diff(kps_norm, axis=0, prepend=kps_norm[0:1])
        acc = np.diff(vel, axis=0, prepend=vel[0:1])
        
        # Relative distances to base joint
        ref = kps_norm[:, 0:1, :]
        rel = kps_norm - ref
        
        # Motion magnitude
        motion_mag = np.linalg.norm(vel, axis=2, keepdims=True)
        
        # Build feature array exactly as in feature_extraction.py
        features = []
        for i in range(target_len):
            frame_features = []
            for j in range(kps_norm.shape[1]):
                frame_features.extend([
                    float(kps_norm[i,j,0]), float(kps_norm[i,j,1]),  # x, y
                    float(vel[i,j,0]), float(vel[i,j,1]),            # vx, vy
                    float(acc[i,j,0]), float(acc[i,j,1]),            # ax, ay
                    float(rel[i,j,0]), float(rel[i,j,1]),            # dx, dy
                    float(motion_mag[i,j,0])                         # motion_mag
                ])
            features.append(frame_features)
            
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Apply the same scaler used during training
        features_reshaped = features.reshape(len(features), -1)
        features_scaled = self.scaler.transform(features_reshaped)
        features = features_scaled.reshape(features.shape)
        
        return features

    def predict_behavior(self, vid_path):
        """End-to-end inference pipeline"""
        try:
            print("\n🔍 Extracting keypoints...")
            keypoints = self.yolo_inference(vid_path)
            if not keypoints:
                return {
                    "status": "error",
                    "message": "No keypoints detected",
                    "class": None,
                    "confidence": 0.0
                }
            
            print("✨ Applying Kalman filtering...")
            smoothed_keypoints = self.apply_kalman_smoothing(keypoints)
            
            print("🔄 Processing features...")
            features = self.process_video_features(smoothed_keypoints)
            if features is None:
                return {
                    "status": "error",
                    "message": "Failed to process features",
                    "class": None,
                    "confidence": 0.0
                }
            
            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            
            print("🤖 Running prediction...")
            pred = self.bilstm_model.predict(features, verbose=0)[0]
            
            pred_class = np.argmax(pred)
            confidence = float(pred[pred_class])
            
            # Get class label from label encoder
            class_label = self.label_encoder.inverse_transform([pred_class])[0]
            
            return {
                "status": "success",
                "class": class_label,
                "class_id": int(pred_class),
                "confidence": confidence,
                "probabilities": {
                    self.label_encoder.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(pred)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "class": None,
                "confidence": 0.0
            }

    def predict_behavior_realtime(self, vid_path, task_id, socketio, sid):
        """Real-time inference pipeline with live keypoints feed"""
        try:
            print("\n🔍 Extracting keypoints in real-time...")
            
            # Process video frame by frame
            keypoints_list = []
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {vid_path}")
            
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = 1 if fps <= 30 else round(fps / 30)
                
                frame_count = 0
                processed_frames = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    if frame_count % frame_interval == 0:
                        processed_frames += 1
                        progress = min(90, (processed_frames / (total_frames // frame_interval)) * 90)
                        
                        # Emit status update
                        socketio.emit('status', {
                            'message': f'Processing frame {processed_frames}...',
                            'progress': progress
                        }, room=sid)
                        
                        # Resize frame for faster inference
                        height, width = frame.shape[:2]
                        original_frame = frame.copy()
                        
                        if width > 320:
                            scale = 320 / width
                            new_width = 320
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Run YOLO inference
                        results = self.yolo_model.predict(frame, conf=0.5, imgsz=320, verbose=False, device='cpu')
                        
                        keypoints = None
                        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                            kps = results[0].keypoints.xy[0].cpu().numpy()
                            # Scale keypoints back to original frame size if we resized
                            if width > 320:
                                scale_x = width / 320
                                scale_y = height / new_height
                                kps = kps * np.array([scale_x, scale_y])
                            keypoints = kps.tolist()
                        
                        keypoints_list.append(keypoints)
                        
                        # Draw keypoints on frame and emit
                        display_frame = original_frame.copy()
                        if keypoints:
                            for i, (x, y) in enumerate(keypoints):
                                if x > 0 and y > 0:  # Valid keypoint
                                    cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                    cv2.putText(display_frame, str(i), (int(x)+5, int(y)-5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # Encode frame as base64 for transmission
                        _, buffer = cv2.imencode('.jpg', display_frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Emit frame with keypoints
                        socketio.emit('frame', {
                            'image': f'data:image/jpeg;base64,{frame_base64}',
                            'keypoints': keypoints,
                            'frame_number': processed_frames
                        }, room=sid)
                        
                        # Small delay to prevent overwhelming the client
                        import time
                        time.sleep(0.01)
                
                print("✨ Applying Kalman filtering...")
                socketio.emit('status', {'message': 'Applying Kalman filtering...', 'progress': 92}, room=sid)
                
                smoothed_keypoints = self.apply_kalman_smoothing(keypoints_list)
                
                print("🔄 Processing features...")
                socketio.emit('status', {'message': 'Processing features...', 'progress': 95}, room=sid)
                
                features = self.process_video_features(smoothed_keypoints)
                if features is None:
                    return {
                        "status": "error",
                        "message": "Failed to process features",
                        "class": None,
                        "confidence": 0.0
                    }
                
                # Add batch dimension
                features = np.expand_dims(features, axis=0)
                
                print("🤖 Running prediction...")
                socketio.emit('status', {'message': 'Running final prediction...', 'progress': 98}, room=sid)
                
                pred = self.bilstm_model.predict(features, verbose=0)[0]
                
                pred_class = np.argmax(pred)
                confidence = float(pred[pred_class])
                
                # Get class label from label encoder
                class_label = self.label_encoder.inverse_transform([pred_class])[0]
                
                return {
                    "status": "success",
                    "class": class_label,
                    "class_id": int(pred_class),
                    "confidence": confidence,
                    "probabilities": {
                        self.label_encoder.inverse_transform([i])[0]: float(p)
                        for i, p in enumerate(pred)
                    }
                }
                
            finally:
                cap.release()
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "class": None,
                "confidence": 0.0
            }

if __name__ == "__main__":
    try:
        print("\n🚀 Initializing behavior detection pipeline...")
        pipeline = DogBehaviorPipeline()
        
        video_path = "BW_blacknwhite/G3_BW.mp4"
        print(f"\n🎥 Processing video: {video_path}")
        
        result = pipeline.predict_behavior(video_path)
        
        if result["status"] == "success":
            print("\n✅ Detection Results:")
            print("=" * 50)
            print(f"Detected Behavior: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\n📊 All Class Probabilities:")
            print("-" * 50)
            for class_name, prob in sorted(result["probabilities"].items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
                print(f"{class_name:15s}: {prob:.2%}")
        else:
            print(f"\n❌ Error: {result['message']}")
            
    except Exception as e:
        print(f"❌ Pipeline Error: {str(e)}")