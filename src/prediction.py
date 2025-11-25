"""
prediction.py
Prediction service for Traffic-Net classification
"""

import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import time

class PredictionService:
    """
    Handles model predictions with caching and performance monitoring
    """
    
    def __init__(self, model_path=None, img_size=(224, 224), model_obj=None):
        self.img_size = img_size
        self.class_names = ['Sparse Traffic', 'Dense Traffic', 'Accident', 'Fire']
        self.model = None
        self.model_path = Path(model_path) if model_path is not None else None
        self.prediction_history = []

        # If a ready model object is provided, use it directly
        if model_obj is not None:
            self.model = model_obj
            print("PredictionService: using provided model object (no load).")
        elif self.model_path is not None:
            self.load_model()
        else:
            raise ValueError("Either model_path or model_obj must be provided to PredictionService.")
    
    def load_model(self):
        """
        Load the trained model
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        print(f"Loading model from {self.model_path}...")

        # Primary: try to load the model normally (no compile to avoid metric issues)
        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
            print("Model loaded successfully (standard load)!")
        except Exception as e:
            # Fallbacks for known Keras serialization issues (e.g. Functional base inside Sequential)
            print(f"Standard load failed: {e}")
            print("Attempting fallback: reconstruct architecture and load weights...")

            # Try to read metadata for num_classes and input size
            num_classes = None
            metadata_path = self.model_path.parent / 'model_metadata.json'
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    num_classes = int(meta.get('num_classes')) if meta.get('num_classes') else None
                    # try to infer img size
                    if meta.get('input_shape') and isinstance(meta.get('input_shape'), (list, tuple)):
                        input_shape = tuple(meta.get('input_shape')[:2])
                        self.img_size = input_shape
                except Exception:
                    num_classes = None

            # Default to 4 classes if unknown
            if num_classes is None:
                num_classes = len(self.class_names) if getattr(self, 'class_names', None) else 4

            # Lazy import to avoid circular imports if any
            try:
                from src.model import TrafficNetModel
            except Exception:
                # try relative import as fallback when running as script
                from model import TrafficNetModel

            # Reconstruct architecture and try to load weights
            try:
                tm = TrafficNetModel(model_path=None, img_size=self.img_size, num_classes=num_classes)
                model = tm.create_model()

                # Choose candidate weight files to try
                candidates = []
                # If provided path is a file with weights-like ext, try it first
                if self.model_path.suffix in ['.h5', '.keras']:
                    candidates.append(self.model_path)

                # common files in the parent models dir
                parent = self.model_path.parent
                for name in ['traffic_net_model.h5', 'best_model.h5', 'traffic_net_model.keras']:
                    p = parent / name
                    if p.exists() and p not in candidates:
                        candidates.append(p)

                loaded = False
                last_err = None
                for cand in candidates:
                    try:
                        print(f"Trying to load weights from {cand}...")
                        model.load_weights(cand)
                        self.model = model
                        loaded = True
                        print(f"Weights loaded from {cand}")
                        break
                    except Exception as le:
                        last_err = le
                        # try by_name as an alternative
                        try:
                            model.load_weights(cand, by_name=True)
                            self.model = model
                            loaded = True
                            print(f"Weights loaded (by_name) from {cand}")
                            break
                        except Exception as le2:
                            last_err = le2
                            print(f"Failed to load weights from {cand}: {le2}")

                if not loaded:
                    # If nothing worked, raise the original exception to surface it
                    raise RuntimeError(f"Fallback weight loading failed. Last error: {last_err}")

            except Exception as e2:
                print(f"Fallback reconstruction failed: {e2}")
                raise e

        # Warm up model with dummy prediction (if model available)
        if self.model is not None:
            try:
                dummy_input = np.random.rand(1, *self.img_size, 3).astype(np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                print("Model warmed up and ready for predictions.")
            except Exception:
                # Non-fatal: warming up may fail for some restored models
                print("Model warming attempt failed but model is loaded.")
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for prediction
        
        Args:
            image_input: Can be path (str/Path), numpy array, or PIL Image
            
        Returns:
            Preprocessed image array
        """
        # Load image based on input type
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                img = np.array(Image.open(image_input))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            raise ValueError("Unsupported image input type")
        
        # Resize
        img_resized = cv2.resize(img, self.img_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    
    def predict(self, image_input, return_all_probs=True):
        """
        Make prediction on a single image
        
        Args:
            image_input: Image to predict (path, array, or PIL Image)
            return_all_probs: Whether to return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess
        img_batch, original_img = self.preprocess_image(image_input)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Inference time
        inference_time = time.time() - start_time
        
        # Prepare result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        
        # Store in history
        self.prediction_history.append({
            'timestamp': result['timestamp'],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time_ms': result['inference_time_ms']
        })
        
        return result
    
    def predict_batch(self, image_inputs, batch_size=32):
        """
        Make predictions on multiple images
        
        Args:
            image_inputs: List of images (paths, arrays, or PIL Images)
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i+batch_size]
            
            # Preprocess batch
            batch_arrays = []
            for img in batch:
                try:
                    img_batch, _ = self.preprocess_image(img)
                    batch_arrays.append(img_batch[0])
                except Exception as e:
                    print(f"Error processing image: {e}")
                    results.append({
                        'error': str(e),
                        'predicted_class': 'Error',
                        'confidence': 0.0
                    })
                    continue
            
            if not batch_arrays:
                continue
            
            # Predict batch
            batch_input = np.array(batch_arrays)
            predictions = self.model.predict(batch_input, verbose=0)
            
            # Process predictions
            for pred in predictions:
                predicted_class_idx = np.argmax(pred)
                predicted_class = self.class_names[predicted_class_idx]
                confidence = float(pred[predicted_class_idx])
                
                results.append({
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': {
                        self.class_names[j]: float(pred[j]) 
                        for j in range(len(self.class_names))
                    }
                })
        
        return results
    
    def predict_with_visualization(self, image_path, save_path=None):
        """
        Make prediction and create visualization
        
        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
            
        Returns:
            Prediction result dictionary
        """
        import matplotlib.pyplot as plt
        
        # Make prediction
        result = self.predict(image_path)
        
        # Load original image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.img_size)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Show image
        ax1.imshow(img_resized)
        ax1.set_title(
            f"Predicted: {result['predicted_class']}\n"
            f"Confidence: {result['confidence']:.2%}",
            fontsize=12, fontweight='bold'
        )
        ax1.axis('off')
        
        # Show probabilities
        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        
        bars = ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('Probability', fontweight='bold')
        ax2.set_title('Class Probabilities', fontweight='bold')
        ax2.set_xlim([0, 1])
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.02, i, f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def get_prediction_statistics(self):
        """
        Get statistics from prediction history
        
        Returns:
            Dictionary with statistics
        """
        if not self.prediction_history:
            return {
                'total_predictions': 0,
                'avg_inference_time_ms': 0,
                'class_distribution': {}
            }
        
        # Calculate statistics
        inference_times = [p['inference_time_ms'] for p in self.prediction_history]
        class_counts = {}
        
        for pred in self.prediction_history:
            class_name = pred['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'avg_inference_time_ms': round(np.mean(inference_times), 2),
            'min_inference_time_ms': round(min(inference_times), 2),
            'max_inference_time_ms': round(max(inference_times), 2),
            'class_distribution': class_counts
        }
        
        return stats
    
    def clear_history(self):
        """
        Clear prediction history
        """
        self.prediction_history = []
        print("Prediction history cleared.")
    
    def export_predictions(self, filepath):
        """
        Export prediction history to JSON file
        
        Args:
            filepath: Path to save predictions
        """
        filepath = Path(filepath)
        
        with open(filepath, 'w') as f:
            json.dump(self.prediction_history, f, indent=4)
        
        print(f"Predictions exported to {filepath}")


class BatchPredictor:
    """
    Optimized batch prediction for high-throughput scenarios
    """
    
    def __init__(self, model_path, batch_size=32):
        self.prediction_service = PredictionService(model_path)
        self.batch_size = batch_size
    
    def predict_directory(self, directory_path, pattern='*.jpg'):
        """
        Predict all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            pattern: Glob pattern for image files
            
        Returns:
            Dictionary mapping filenames to predictions
        """
        directory = Path(directory_path)
        image_paths = list(directory.glob(pattern))
        
        if not image_paths:
            print(f"No images found matching pattern: {pattern}")
            return {}
        
        print(f"Processing {len(image_paths)} images...")
        
        results = {}
        start_time = time.time()
        
        # Process in batches
        predictions = self.prediction_service.predict_batch(
            image_paths, 
            batch_size=self.batch_size
        )
        
        # Map predictions to filenames
        for img_path, pred in zip(image_paths, predictions):
            results[img_path.name] = pred
        
        total_time = time.time() - start_time
        print(f"Processed {len(image_paths)} images in {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_paths)*1000:.2f}ms")
        
        return results
    
    def predict_and_save_results(self, directory_path, output_path):
        """
        Predict all images and save results to file
        
        Args:
            directory_path: Path to directory containing images
            output_path: Path to save results JSON
        """
        results = self.predict_directory(directory_path)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {output_path}")


def predict_single_image_cli(model_path, image_path):
    """CLI function for single image prediction using a model path"""
    predictor = PredictionService(model_path)
    result = predictor.predict_with_visualization(image_path)
    _print_result(result, image_path)


def predict_image_with_weights_cli(weights_path, image_path, img_size=(224, 224), num_classes=4):
    """CLI entry to reconstruct architecture and load weights then predict"""
    # Reconstruct architecture using TrafficNetModel
    try:
        from src.model import TrafficNetModel
    except Exception:
        from model import TrafficNetModel

    tm = TrafficNetModel(model_path=None, img_size=img_size, num_classes=num_classes)
    model = tm.create_model()

    print(f"Loading weights from {weights_path}...")
    model.load_weights(weights_path)

    # Use PredictionService with provided model object
    predictor = PredictionService(model_path=None, img_size=img_size, model_obj=model)
    result = predictor.predict_with_visualization(image_path)
    _print_result(result, image_path)


def _print_result(result, image_path):
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference Time: {result['inference_time_ms']}ms")
    print("\nAll Probabilities:")
    for class_name, prob in result.get('all_probabilities', {}).items():
        print(f"  {class_name}: {prob:.4f}")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prediction CLI for Traffic-Net')
    parser.add_argument('--model', help='Path to a saved model to load (keras/savedmodel)')
    parser.add_argument('--weights', help='Path to a weights file to load into reconstructed model')
    parser.add_argument('--image', help='Path to image to predict')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224], help='Input image size (width height)')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes for reconstructed model')

    args = parser.parse_args()

    if args.model and args.image:
        if Path(args.model).exists():
            predict_single_image_cli(args.model, args.image)
        else:
            print(f"Model not found at {args.model}")
    elif args.weights and args.image:
        if Path(args.weights).exists():
            predict_image_with_weights_cli(args.weights, args.image, img_size=tuple(args.img_size), num_classes=args.num_classes)
        else:
            print(f"Weights file not found at {args.weights}")
    else:
        # Fallback: try default model path for quick test
        default_model = Path("../models/traffic_net_model.keras")
        if default_model.exists():
            print("Testing PredictionService with default model...")
            predictor = PredictionService(str(default_model))
            print("\nPrediction service initialized!")
            print(f"Model: {default_model}")
            print(f"Classes: {predictor.class_names}")
            stats = predictor.get_prediction_statistics()
            print(f"\nStatistics: {stats}")
        else:
            print("No model or weights provided and default model not found.")