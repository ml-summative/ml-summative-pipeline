"""
app.py
Flask API with Web UI for Traffic-Net Classification
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import requests
from pathlib import Path
import json
from datetime import datetime
import time
import threading

# Import custom modules
import sys
sys.path.append('src')
from prediction import PredictionService
from model import TrafficNetModel
from preprocessing import ImagePreprocessor, DataPipeline

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('data/retrain').mkdir(parents=True, exist_ok=True)

# Global variables
MODEL_PATH = 'models/traffic_net_model.h5'
predictor = None
model_handler = None
preprocessor = ImagePreprocessor()
data_pipeline = DataPipeline()

# Model status
model_status = {
    'status': 'offline',
    'uptime_start': None,
    'total_predictions': 0,
    'last_prediction_time': None,
    'is_training': False
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def ensure_model_from_url(target_path='models/traffic_net_model.h5'):
    model_path = Path(target_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1000:
        # already present and non-empty
        return True

    url = os.environ.get('MODEL_URL')
    if not url:
        print("No MODEL_URL set; skipping model download.")
        return False

    headers = {}
    token = os.environ.get('MODEL_AUTH_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    print(f"Downloading model from {url} ...")
    with requests.get(url, stream=True, headers=headers, timeout=300) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Model saved to {model_path}")
    return True

ensure_model_from_url()

def initialize_model():
    """Initialize the prediction service"""
    global predictor, model_status
    
    try:
        predictor = PredictionService(MODEL_PATH)
        model_status['status'] = 'online'
        model_status['uptime_start'] = datetime.now()
        print("✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_status['status'] = 'error'
        return False

# Initialize on startup
initialize_model()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model and system status"""
    status = model_status.copy()
    
    if status['uptime_start']:
        uptime_seconds = (datetime.now() - status['uptime_start']).total_seconds()
        status['uptime_seconds'] = int(uptime_seconds)
    
    # Get prediction statistics if available
    if predictor:
        stats = predictor.get_prediction_statistics()
        status['prediction_stats'] = stats
    
    return jsonify(status)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    global model_status
    
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Timing: start request handling
        req_start = time.time()

        # Make prediction (measure separately)
        pred_start = time.time()
        result = predictor.predict(filepath)
        pred_end = time.time()

        # Update status
        model_status['total_predictions'] += 1
        model_status['last_prediction_time'] = datetime.now().isoformat()

        # Add file info to result
        result['filename'] = filename
        result['file_path'] = filepath

        # Record timings (ms)
        req_end = time.time()
        timings = {
            'request_total_ms': round((req_end - req_start) * 1000, 2),
            'predict_ms': round((pred_end - pred_start) * 1000, 2),
            'overhead_ms': round(((req_end - req_start) - (pred_end - pred_start)) * 1000, 2)
        }
        if isinstance(result, dict):
            result['timings'] = timings

        # Log timing info
        print(f"[TIMING] /api/predict - total={timings['request_total_ms']}ms predict={timings['predict_ms']}ms overhead={timings['overhead_ms']}ms")

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Make predictions on multiple images"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict
                result = predictor.predict(filepath)
                result['filename'] = filename
                results.append(result)
                
                model_status['total_predictions'] += 1
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({
        'total_processed': len(results),
        'results': results
    })

@app.route('/api/upload/retrain', methods=['POST'])
def upload_for_retraining():
    """Upload images for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    class_label = request.form.get('class_label', 'unlabeled')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    saved_files = []
    retrain_dir = Path('data/retrain') / class_label
    retrain_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = retrain_dir / filename
                file.save(str(filepath))
                saved_files.append(str(filepath))
            
            except Exception as e:
                print(f"Error saving {file.filename}: {e}")
    
    return jsonify({
        'message': f'Uploaded {len(saved_files)} images',
        'files_saved': len(saved_files),
        'class_label': class_label
    })

@app.route('/api/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    global model_status, predictor
    
    if model_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Get training parameters
    data = request.get_json() or {}
    epochs = data.get('epochs', 30)
    learning_rate = data.get('learning_rate', 1e-4)
    
    def retrain_model():
        global model_status, predictor
        
        try:
            model_status['is_training'] = True
            model_status['status'] = 'retraining'
            
            print("Starting retraining process...")
            
            # Merge retraining data with existing training data
            data_pipeline.merge_retraining_data()
            
            # Create data generators
            train_gen, val_gen, test_gen = preprocessor.create_data_generators(
                'data/train',
                'data/test',
                batch_size=32
            )
            
            # Load existing model
            model_handler = TrafficNetModel(model_path=MODEL_PATH)

            # Ensure model output size matches training data classes
            try:
                train_classes = len(train_gen.class_indices) if hasattr(train_gen, 'class_indices') else None
            except Exception:
                train_classes = None

            if train_classes and train_classes != model_handler.num_classes:
                print(f"Warning: training data has {train_classes} classes but model expects {model_handler.num_classes}. Rebuilding model to match training classes...")
                # Rebuild model with correct number of classes and try to load weights by name
                model_handler.num_classes = train_classes
                new_model = model_handler.create_model()
                try:
                    new_model.load_weights(MODEL_PATH, by_name=True)
                    print("Loaded existing weights into rebuilt model (by_name).")
                except Exception as e:
                    print(f"Could not load weights into rebuilt model: {e}")
                model_handler.model = new_model
            
            # Retrain
            history = model_handler.retrain(
                train_gen,
                val_gen,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            # Save retrained model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_path = f'models/traffic_net_model_{timestamp}.h5'
            model_handler.save_model(new_model_path)
            
            # Reload predictor with new model
            predictor = PredictionService(new_model_path)
            
            model_status['is_training'] = False
            model_status['status'] = 'online'
            model_status['last_retrain'] = datetime.now().isoformat()
            
            print("✓ Retraining completed successfully!")
        
        except Exception as e:
            print(f"✗ Retraining error: {e}")
            model_status['is_training'] = False
            model_status['status'] = 'error'
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_model)
    thread.start()
    
    return jsonify({
        'message': 'Retraining started',
        'epochs': epochs,
        'learning_rate': learning_rate
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model evaluation metrics"""
    metadata_path = Path('models/model_metadata.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    else:
        return jsonify({'error': 'Metrics not available'}), 404

@app.route('/api/visualizations/data', methods=['GET'])
def get_visualization_data():
    """Get data for visualizations"""
    # Get dataset statistics
    train_stats = preprocessor.get_dataset_statistics('data/train')
    test_stats = preprocessor.get_dataset_statistics('data/test')
    
    # Get prediction statistics
    pred_stats = predictor.get_prediction_statistics() if predictor else {}
    
    return jsonify({
        'train_stats': train_stats,
        'test_stats': test_stats,
        'prediction_stats': pred_stats
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use the PORT env var (Render sets PORT) or fall back to 5000
    port = int(os.environ.get('PORT') or os.environ.get('RENDER_EXTERNAL_PORT') or 5000)
    print("\n" + "="*60)
    print("Traffic-Net API Server")
    print("="*60)
    print(f"Model Status: {model_status['status']}")
    print(f"Server starting on http://0.0.0.0:{port}")
    print("="*60 + "\n")

    # NOTE: In production (Render) prefer running with a WSGI server such as Gunicorn or Waitress.
    # For quick local/debug runs we still allow starting the Flask app, but disable debug mode.
    app.run(host='0.0.0.0', port=port, debug=False)