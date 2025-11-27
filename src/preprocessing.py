"""
preprocessing.py
Data preprocessing module for Traffic-Net classification
"""

import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from PIL import Image
import os

class ImagePreprocessor:
    """
    Handles all image preprocessing operations
    """
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        # Use human-friendly labels here; directory names are inferred from these
        self.class_names = ['Sparse Traffic', 'Dense Traffic', 'Accident', 'Fire']
        
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
            if img is None:
                # Try with PIL
                img = Image.open(image_path)
                img = np.array(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            img = image_path
        else:
            raise ValueError("Input must be image path or numpy array")
        
        # Resize
        img_resized = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess multiple images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Batch of preprocessed images
        """
        images = []
        for path in image_paths:
            try:
                img = self.preprocess_single_image(path)
                images.append(img[0])  # Remove batch dimension
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if len(images) == 0:
            return None
        
        return np.array(images)
    
    def augment_image(self, image, augmentation_params=None):
        """
        Apply data augmentation to an image
        
        Args:
            image: Input image array
            augmentation_params: Dictionary of augmentation parameters
            
        Returns:
            Augmented image
        """
        if augmentation_params is None:
            augmentation_params = {
                'rotation_range': 20,
                'width_shift_range': 0.2,
                'height_shift_range': 0.2,
                'zoom_range': 0.15,
                'horizontal_flip': True
            }
        
        datagen = ImageDataGenerator(**augmentation_params)
        
        # Reshape for ImageDataGenerator
        img_reshaped = image.reshape((1,) + image.shape)
        
        # Generate augmented image
        aug_iter = datagen.flow(img_reshaped, batch_size=1)
        augmented_img = next(aug_iter)[0]
        
        return augmented_img
    
    def create_data_generators(self, train_dir, test_dir, 
                               batch_size=32, validation_split=0.2):
        """
        Create data generators for training and testing
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory
            batch_size: Batch size for generators
            validation_split: Fraction of training data to use for validation
            
        Returns:
            train_gen, val_gen, test_gen
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Test data (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen
    
    def save_uploaded_images(self, uploaded_files, save_dir, class_name=None):
        """
        Save uploaded images to appropriate directories for retraining
        
        Args:
            uploaded_files: List of uploaded file objects
            save_dir: Base directory to save images
            class_name: Optional class name (if known)
            
        Returns:
            List of saved file paths
        """
        save_dir = Path(save_dir)
        saved_paths = []
        
        for i, file in enumerate(uploaded_files):
            try:
                # Determine save path
                if class_name:
                    # Normalize class directory names to lowercase with underscores
                    dir_name = class_name.lower().replace(' ', '_')
                    class_dir = save_dir / dir_name
                else:
                    class_dir = save_dir / 'unlabeled'
                
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename
                filename = f"upload_{i}_{file.filename}"
                save_path = class_dir / filename
                
                # Save file
                if hasattr(file, 'save'):
                    file.save(str(save_path))
                else:
                    # Handle different file object types
                    with open(save_path, 'wb') as f:
                        f.write(file.read())
                
                saved_paths.append(str(save_path))
                
            except Exception as e:
                print(f"Error saving file {file.filename}: {e}")
                continue
        
        return saved_paths
    
    def validate_image(self, image_path):
        """
        Validate if an image is valid and can be processed
        
        Args:
            image_path: Path to image file
            
        Returns:
            Boolean indicating if image is valid
        """
        try:
            img = Image.open(image_path)
            img.verify()
            
            # Check if image can be loaded
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            # Check dimensions
            h, w = img.shape[:2]
            if h < 50 or w < 50:
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error for {image_path}: {e}")
            return False
    
    def get_dataset_statistics(self, data_dir):
        """
        Calculate statistics about the dataset
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        data_dir = Path(data_dir)

        # If the requested path doesn't exist, fall back to project-level `data/`
        if not data_dir.exists():
            fallback = Path('data')
            if fallback.exists():
                data_dir = fallback
            else:
                # Nothing to report
                return stats
        stats = {
            'total_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'avg_size': None
        }
        
        for class_name in self.class_names:
            # Prefer underscored lowercase directory names (e.g., 'sparse_traffic')
            dir_name = class_name.lower().replace(' ', '_')
            class_path = data_dir / dir_name

            # Fallback: some datasets may use human-friendly names as directories
            if not class_path.exists():
                class_path = data_dir / class_name
            # If not found at top-level, search recursively for matching folders
            found_dirs = []
            if class_path.exists():
                found_dirs.append(class_path)
            else:
                # Search for any directory matching the normalized or human name
                target_names = {dir_name, class_name.lower()}
                for p in data_dir.rglob('*'):
                    if p.is_dir() and p.name.lower() in target_names:
                        found_dirs.append(p)

            # Aggregate counts across all matching directories (handles train/test subfolders)
            total_count = 0
            sampled_sizes = []
            for d in found_dirs:
                imgs = list(d.glob('*.jpg')) + list(d.glob('*.png')) + list(d.glob('*.jpeg'))
                total_count += len(imgs)
                for img_path in imgs[:10]:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            sampled_sizes.append(img.shape[:2])
                    except:
                        continue

            if total_count > 0:
                stats['class_distribution'][class_name] = total_count
                stats['total_images'] += total_count
                stats['image_sizes'].extend(sampled_sizes)
        
        if stats['image_sizes']:
            avg_h = np.mean([s[0] for s in stats['image_sizes']])
            avg_w = np.mean([s[1] for s in stats['image_sizes']])
            stats['avg_size'] = (int(avg_h), int(avg_w))
        
        return stats


class DataPipeline:
    """
    Complete data pipeline for managing training data
    """
    
    def __init__(self, base_dir='../data'):
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / 'train'
        self.test_dir = self.base_dir / 'test'
        self.retrain_dir = self.base_dir / 'retrain'
        self.preprocessor = ImagePreprocessor()
        
        # Create directories
        for dir_path in [self.train_dir, self.test_dir, self.retrain_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_retraining_data(self, uploaded_images, class_labels=None):
        """
        Prepare uploaded images for retraining
        
        Args:
            uploaded_images: List of uploaded image files
            class_labels: Optional list of class labels for each image
            
        Returns:
            Path to prepared retraining data
        """
        # Save uploaded images
        saved_paths = []
        
        for i, img in enumerate(uploaded_images):
            if class_labels and i < len(class_labels):
                class_name = class_labels[i]
            else:
                class_name = 'unlabeled'
            
            paths = self.preprocessor.save_uploaded_images(
                [img], 
                self.retrain_dir, 
                class_name
            )
            saved_paths.extend(paths)
        
        return saved_paths
    
    def merge_retraining_data(self):
        """
        Merge retraining data with existing training data
        """
        # Move images from retrain_dir to train_dir using normalized directory names
        for class_name in self.preprocessor.class_names:
            dir_name = class_name.lower().replace(' ', '_')
            retrain_class_dir = self.retrain_dir / dir_name
            train_class_dir = self.train_dir / dir_name

            # Also support legacy folders that used the human-friendly label
            if not retrain_class_dir.exists():
                retrain_class_dir = self.retrain_dir / class_name
            if not train_class_dir.exists():
                train_class_dir = self.train_dir / class_name

            if retrain_class_dir.exists():
                train_class_dir.mkdir(parents=True, exist_ok=True)

                for img_path in retrain_class_dir.glob('*'):
                    if img_path.is_file():
                        # Move to training directory
                        new_path = train_class_dir / img_path.name
                        img_path.rename(new_path)
    
    def split_data(self, source_dir, train_ratio=0.8):
        """
        Split data into train and test sets
        
        Args:
            source_dir: Source directory containing class folders
            train_ratio: Ratio of data to use for training
        """
        source_dir = Path(source_dir)
        
        for class_name in self.preprocessor.class_names:
            class_path = source_dir / class_name
            if not class_path.exists():
                continue
            
            # Get all images
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            np.random.shuffle(images)
            
            # Split
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            # Create directories
            (self.train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (self.test_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for img_path in train_images:
                new_path = self.train_dir / class_name / img_path.name
                if not new_path.exists():
                    img_path.rename(new_path)
            
            for img_path in test_images:
                new_path = self.test_dir / class_name / img_path.name
                if not new_path.exists():
                    img_path.rename(new_path)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    print("Image Preprocessor initialized")
    print(f"Image size: {preprocessor.img_size}")
    print(f"Classes: {preprocessor.class_names}")
    
    # Test data pipeline
    pipeline = DataPipeline()
    print("\nData Pipeline initialized")
    print(f"Train directory: {pipeline.train_dir}")
    print(f"Test directory: {pipeline.test_dir}")