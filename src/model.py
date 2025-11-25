"""
model.py
Model creation, training, and retraining module for Traffic-Net
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pickle

class TrafficNetModel:
    """
    Traffic-Net classification model with training and retraining capabilities
    """
    
    def __init__(self, model_path=None, img_size=(224, 224), num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['Sparse Traffic', 'Dense Traffic', 'Accident', 'Fire']
        self.model = None
        self.history = None
        self.model_dir = Path('../models')
        self.model_dir.mkdir(exist_ok=True)
        
        if model_path:
            self.load_model(model_path)
    
    def create_model(self, base_architecture='mobilenetv2'):
        """
        Create a new model using transfer learning
        
        Args:
            base_architecture: Base model to use ('mobilenetv2', 'efficientnet', 'resnet50')
            
        Returns:
            Compiled Keras model
        """
        input_shape = self.img_size + (3,)
        
        # Select base model
        if base_architecture.lower() == 'mobilenetv2':
            base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_architecture.lower() == 'efficientnet':
            base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        elif base_architecture.lower() == 'resnet50':
            base_model = ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unknown architecture: {base_architecture}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.base_model = base_model
        return self.model
    
    def train(self, train_generator, validation_generator, 
              epochs=50, callbacks=None, fine_tune=True):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
            fine_tune: Whether to perform fine-tuning after initial training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.create_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        print("=" * 60)
        print("PHASE 1: Training with frozen base model")
        print("=" * 60)
        
        # Phase 1: Train with frozen base
        history_phase1 = self.model.fit(
            train_generator,
            epochs=min(20, epochs // 2),
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (if enabled)
        if fine_tune and hasattr(self, 'base_model'):
            print("\n" + "=" * 60)
            print("PHASE 2: Fine-tuning entire model")
            print("=" * 60)
            
            # Unfreeze base model
            self.base_model.trainable = True
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')
                ]
            )
            
            # Continue training
            history_phase2 = self.model.fit(
                train_generator,
                epochs=epochs - min(20, epochs // 2),
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = self._combine_histories(history_phase1, history_phase2)
        else:
            self.history = history_phase1
        
        return self.history
    
    def retrain(self, train_generator, validation_generator, 
                epochs=30, learning_rate=1e-4):
        """
        Retrain an existing model with new data
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of epochs for retraining
            learning_rate: Learning rate for retraining
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("No model loaded. Create or load a model first.")
        
        print("=" * 60)
        print("RETRAINING MODEL")
        print("=" * 60)
        
        # Unfreeze all layers
        for layer in self.model.layers:
            layer.trainable = True
        
        # Recompile with specified learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Get callbacks for retraining
        callbacks = self._get_default_callbacks(prefix='retrain')
        
        # Train
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate(self, test_generator):
        """
        Evaluate model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded.")
        
        # Evaluate
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions for detailed metrics
        test_generator.reset()
        y_pred_probs = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3],
            'auc': results[4],
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'predictions': y_pred.tolist(),
            'true_labels': y_true.tolist(),
            'probabilities': y_pred_probs.tolist()
        }
        
        return metrics
    
    def save_model(self, filepath=None, save_metadata=True):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save model (default: models/traffic_net_model.h5)
            save_metadata: Whether to save metadata alongside model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if filepath is None:
            filepath = self.model_dir / 'traffic_net_model.h5'
        else:
            filepath = Path(filepath)
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Save metadata
        if save_metadata:
            metadata_path = filepath.parent / 'model_metadata.json'
            metadata = {
                'model_name': 'Traffic-Net Classifier',
                'version': self._get_version(),
                'created_at': datetime.now().isoformat(),
                'architecture': 'MobileNetV2 + Custom Layers',
                'input_shape': list(self.img_size) + [3],
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'model_path': str(filepath)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath: Path to saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        # Try to load metadata
        metadata_path = filepath.parent / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Metadata loaded: {metadata.get('version', 'Unknown version')}")
    
    def _get_default_callbacks(self, prefix=''):
        """
        Get default training callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.model_dir / f'{prefix}best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def _combine_histories(self, hist1, hist2):
        """
        Combine two training histories
        """
        combined = keras.callbacks.History()
        combined.history = {}
        
        for key in hist1.history.keys():
            combined.history[key] = hist1.history[key] + hist2.history[key]
        
        return combined
    
    def _get_version(self):
        """
        Get model version based on existing models
        """
        existing_models = list(self.model_dir.glob('traffic_net_model*.h5'))
        return f"v{len(existing_models) + 1}.0"
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        if self.model is None:
            return "No model loaded."
        
        from io import StringIO
        import sys
        
        # Capture summary
        old_stdout = sys.stdout
        sys.stdout = summary_buffer = StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return summary_buffer.getvalue()


class ModelManager:
    """
    Manages multiple models and versioning
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.current_model = None
    
    def list_models(self):
        """
        List all available models
        """
        models = list(self.model_dir.glob('*.h5'))
        return [m.name for m in models]
    
    def load_latest_model(self):
        """
        Load the most recent model
        """
        models = list(self.model_dir.glob('traffic_net_model*.h5'))
        if not models:
            raise FileNotFoundError("No models found.")
        
        latest_model = max(models, key=lambda p: p.stat().st_mtime)
        self.current_model = TrafficNetModel(model_path=latest_model)
        return self.current_model
    
    def create_model_checkpoint(self, model, name=None):
        """
        Create a checkpoint of the current model
        """
        if name is None:
            name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        
        checkpoint_path = self.model_dir / name
        model.save_model(checkpoint_path)
        return checkpoint_path


if __name__ == "__main__":
    # Test model creation
    print("Testing TrafficNetModel...")
    
    model_handler = TrafficNetModel()
    model = model_handler.create_model()
    
    print("\nModel created successfully!")
    print(model_handler.get_model_summary())
    
    print(f"\nClass names: {model_handler.class_names}")
    print(f"Number of classes: {model_handler.num_classes}")