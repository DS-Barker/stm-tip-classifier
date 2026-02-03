"""
Data Generators for STM Tip Classification
==========================================

Provides data loading and augmentation pipelines for training CNN classifiers
on STM images.

Features:
- Loads images from directory structure (good/bad subdirectories)
- Real-time data augmentation (horizontal/vertical flips)
- Batch creation for efficient GPU training
- Automatic grayscale conversion for STM topography images

Data Organization:
    data/surface_name/processed/
    ├── training/
    │   ├── Good/
    │   └── Bad/
    └── validation/
        ├── Good/
        └── Bad/

Usage:
    from src.data_processing.generators import train_val_generators
    
    train_gen, val_gen = train_val_generators(
        training_dir='data/bsi/processed/training',
        validation_dir='data/bsi/processed/validation',
        batch_size=32,
        target_size=(140, 140)
    )
    model.fit(train_gen, epochs=30, validation_data=val_gen)

Author: Dylan S. Barker, University of Leeds, 2024
"""

from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def train_val_generators(training_dir, 
                         validation_dir, 
                         batch_size=32, 
                         target_size=(140, 140)):
    """
    Creates training and validation data generators for STM tip classification.
    
    The training generator applies data augmentation (flips) to prevent overfitting,
    while the validation generator only rescales images for consistent evaluation.
    
    Args:
        training_dir (Path or str): Directory containing training data.
            Expected subdirectories: 'Good/' and 'Bad/'
            
        validation_dir (Path or str): Directory containing validation data.
            Expected subdirectories: 'Good/' and 'Bad/'
            
        batch_size (int): Number of images per batch.
            Larger values utilize GPU better but require more memory.
            Typical values: 16, 32, 64
            Default: 32
            
        target_size (tuple): Target image dimensions (height, width) in pixels.
            Images will be resized to this size if they don't match.
            For thesis Si(111)-7×7 models: (700, 700)
            For quick experiments: (140, 140)
            Default: (140, 140)
    
    Returns:
        tuple: (train_generator, validation_generator)
            - train_generator: ImageDataGenerator with augmentation
            - validation_generator: ImageDataGenerator without augmentation
            
    Example:
        >>> train_gen, val_gen = train_val_generators(
        ...     training_dir='data/bsi/processed/training',
        ...     validation_dir='data/bsi/processed/validation',
        ...     batch_size=32,
        ...     target_size=(700, 700)
        ... )
        >>> 
        >>> # Use with model training
        >>> history = model.fit(
        ...     train_gen,
        ...     epochs=100,
        ...     validation_data=val_gen
        ... )
        
    Raises:
        FileNotFoundError: If directories don't exist
        ValueError: If directory structure is incorrect (missing Good/Bad subdirs)
        
    Notes:
        - Training images are augmented with horizontal and vertical flips
        - Both generators rescale pixel values from [0, 255] to [0, 1]
        - Images are converted to grayscale (single channel)
        - Class labels are inferred from subdirectory names
        - Generators infinite loop; use steps_per_epoch in model.fit()
    """
    # Convert to Path objects for better path handling
    training_dir = Path(training_dir)
    validation_dir = Path(validation_dir)
    
    # Validate directories exist
    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {training_dir}\n"
            f"Please create directory structure with 'Good' and 'Bad' subdirectories."
        )
    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {validation_dir}\n"
            f"Please create directory structure with 'Good' and 'Bad' subdirectories."
        )
    
    # Training data generator with augmentation
    # -----------------------------------------
    # Augmentation strategies:
    # - horizontal_flip: Surface features symmetric in x-direction
    # - vertical_flip: Surface features symmetric in y-direction
    # - rescale: Normalize pixel values to [0, 1] for stable training
    #
    # NOT used (would break physical meaning):
    # - rotation_range: Would break lattice orientation
    # - zoom_range: Would change atomic spacing
    # - shear_range: Would distort lattice
    # - brightness/contrast: STM images are quantitative topography
    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,        # Normalize pixel values [0, 255] → [0, 1]
        horizontal_flip=True,        # Random horizontal flips
        vertical_flip=True,          # Random vertical flips
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(training_dir),
        batch_size=batch_size,
        class_mode='binary',         # Binary classification: good (0) vs bad (1)
        target_size=target_size,     # Resize images to this size
        color_mode='grayscale',      # STM images are single-channel topography
        shuffle=True,                # Shuffle batches each epoch
        seed=42                      # Reproducible shuffling
    )

    # Validation data generator (no augmentation)
    # -------------------------------------------
    # Validation set should remain constant across epochs for fair comparison.
    # Only rescaling is applied - no augmentation.
    
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0         # Same normalization as training
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=str(validation_dir),
        batch_size=batch_size,
        class_mode='binary',
        target_size=target_size,
        color_mode='grayscale',
        shuffle=False,               # Don't shuffle validation for reproducibility
        seed=42
    )
    
    # Print generator info for debugging
    print(f"\nData Generators Created:")
    print(f"Training samples: {train_generator.n}")
    print(f"Validation samples: {validation_generator.n}")
    print(f"Batch size: {batch_size}")
    print(f"Target size: {target_size}")
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Steps per epoch: {train_generator.n // batch_size}\n")
    
    return train_generator, validation_generator


def create_test_generator(test_dir, batch_size=32, target_size=(140, 140)):
    """
    Creates a test data generator for final model evaluation.
    
    Similar to validation generator but separate for final testing on
    completely unseen data.
    
    Args:
        test_dir (Path or str): Directory containing test data
        batch_size (int): Number of images per batch
        target_size (tuple): Target image dimensions
        
    Returns:
        ImageDataGenerator: Test data generator
        
    Example:
        >>> test_gen = create_test_generator('data/bsi/processed/test')
        >>> test_loss, test_acc = model.evaluate(test_gen)
    """
    test_dir = Path(test_dir)
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        batch_size=batch_size,
        class_mode='binary',
        target_size=target_size,
        color_mode='grayscale',
        shuffle=False  # Keep test order consistent
    )
    
    print(f"\nTest Generator Created:")
    print(f"Test samples: {test_generator.n}")
    print(f"Class indices: {test_generator.class_indices}\n")
    
    return test_generator


def get_generator_stats(generator):
    """
    Extract statistics from a data generator for analysis.
    
    Args:
        generator: Keras ImageDataGenerator
        
    Returns:
        dict: Statistics including class distribution, batch info
        
    Example:
        >>> stats = get_generator_stats(train_generator)
        >>> print(f"Class balance: {stats['class_distribution']}")
    """
    stats = {
        'total_samples': generator.n,
        'batch_size': generator.batch_size,
        'num_batches': generator.n // generator.batch_size,
        'class_indices': generator.class_indices,
        'target_size': generator.target_size,
        'class_distribution': {}
    }
    
    # Calculate class distribution
    for class_name, class_idx in generator.class_indices.items():
        count = sum(generator.classes == class_idx)
        stats['class_distribution'][class_name] = count
    
    return stats


# Augmentation presets for different use cases
AUGMENTATION_PRESETS = {
    'minimal': {
        'horizontal_flip': True,
        'vertical_flip': True,
        'description': 'Safe augmentations that preserve STM image properties'
    },
    'none': {
        'description': 'No augmentation, only rescaling (for validation/test)'
    }
}


def create_custom_generator(directory, 
                            augmentation='minimal',
                            batch_size=32,
                            target_size=(140, 140)):
    """
    Create a generator with custom augmentation preset.
    
    Args:
        directory: Data directory
        augmentation (str): Preset name ('minimal' or 'none')
        batch_size (int): Batch size
        target_size (tuple): Image dimensions
        
    Returns:
        ImageDataGenerator: Configured generator
    """
    if augmentation == 'minimal':
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            horizontal_flip=True,
            vertical_flip=True
        )
    elif augmentation == 'none':
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    else:
        raise ValueError(f"Unknown augmentation preset: {augmentation}")
    
    generator = datagen.flow_from_directory(
        directory=str(directory),
        batch_size=batch_size,
        class_mode='binary',
        target_size=target_size,
        color_mode='grayscale'
    )
    
    return generator
