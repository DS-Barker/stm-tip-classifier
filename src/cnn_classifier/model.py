"""
CNN Model Architecture for STM Tip Classification
==================================================

Defines convolutional neural network architectures for automated classification
of STM probe tip quality from atomic-resolution images.

The model distinguishes between "good" (sharp, atomic resolution) and "bad" 
(blunt, double tip) probe tips, enabling automated tip preparation.

Architecture:
- Convolutional base: Extracts features from STM images
- Dense classifier: Maps features to binary classification (good/bad)
- Variable depth: Configurable layers for hyperparameter search

Performance: 96% accuracy on Si(111)-7×7, 90% on B:Si

Usage:
    from src.cnn_classifier.model import build_model
    
    model = build_model(
        num_conv_layers=5,
        starting_conv_filters=20,
        filter_multiplier=2,
        num_dense_layers=3,
        starting_dense_neurons=32
    )
    model.fit(train_data, epochs=100, validation_data=val_data)

Author: Dylan S. Barker, University of Leeds, 2024
"""

import tensorflow as tf
from tensorflow import keras


def build_model(num_conv_layers, 
                starting_conv_filters, 
                filter_multiplier,
                num_dense_layers, 
                starting_dense_neurons, 
                conv_kernel_size=(3, 3), 
                pool_size=(2, 2),
                dropout_rate=0.5, 
                input_shape=(140, 140, 1)):
    """
    Builds a CNN model with configurable convolutional and dense layer architecture.
    
    The model follows a standard CNN design pattern:
    - Convolutional base for feature extraction
    - Global pooling or flattening
    - Dense layers for classification
    - Binary output with sigmoid activation
    
    Args:
        num_conv_layers (int): Number of convolutional blocks (Conv2D + MaxPooling2D).
            Typical values: 3-5 for STM images.
            
        starting_conv_filters (int): Number of filters in the first convolutional layer.
            Subsequent layers multiply by filter_multiplier.
            Typical values: 16, 20, 32
            
        filter_multiplier (int): Factor by which filters increase in each conv layer.
            filter_multiplier=2 gives: [16, 32, 64, 128, ...]
            Typical value: 2
            
        num_dense_layers (int): Number of fully connected layers before output.
            Does not include the final binary output layer.
            Typical values: 1-3
            
        starting_dense_neurons (int): Number of neurons in the first dense layer.
            Subsequent layers double in size.
            Typical values: 32, 64, 128
            
        conv_kernel_size (tuple): Size of convolutional kernels. 
            Default: (3, 3)
            
        pool_size (tuple): Size of max pooling windows.
            Default: (2, 2)
            
        dropout_rate (float): Dropout probability for dense layers (0.0-1.0).
            Higher values provide more regularization.
            Default: 0.5
            
        input_shape (tuple): Shape of input images (height, width, channels).
            For grayscale STM images: (height, width, 1)
            Default: (140, 140, 1)
        
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
            - Loss: binary_crossentropy
            - Optimizer: adam
            - Metrics: accuracy
    
    Example:
        >>> # Build a 5-layer CNN with progressive filters [20, 40, 80, 160, 320]
        >>> # and 3 dense layers [32, 64, 128]
        >>> model = build_model(
        ...     num_conv_layers=5,
        ...     starting_conv_filters=20,
        ...     filter_multiplier=2,
        ...     num_dense_layers=3,
        ...     starting_dense_neurons=32,
        ...     input_shape=(700, 700, 1)
        ... )
        >>> model.summary()
        
    Notes:
        - All convolutional layers use ReLU activation
        - All dense layers use ReLU activation with dropout
        - Final output layer uses sigmoid for binary classification
        - Model is compiled with Adam optimizer and binary crossentropy loss
        
    Raises:
        ValueError: If parameters result in invalid architecture
        (e.g., negative filter counts, incompatible input/output sizes)
    """
    layers = []
    
    # Build convolutional base with variable depth
    current_filters = starting_conv_filters
    
    for i in range(num_conv_layers):
        if i == 0:
            # First convolutional layer needs input_shape specification
            layers.append(
                keras.layers.Conv2D(
                    current_filters, 
                    conv_kernel_size, 
                    activation='relu', 
                    input_shape=input_shape,
                    padding='same',
                    name=f'conv_{i+1}'
                )
            )
        else:
            # Subsequent layers don't need input_shape
            layers.append(
                keras.layers.Conv2D(
                    current_filters, 
                    conv_kernel_size, 
                    activation='relu',
                    padding='same',
                    name=f'conv_{i+1}'
                )
            )
        
        # Add max pooling after each convolution
        layers.append(
            keras.layers.MaxPooling2D(
                pool_size,
                name=f'pool_{i+1}'
            )
        )
        
        # Increase filter count for next layer (geometric progression)
        current_filters *= filter_multiplier
    
    # Flatten feature maps before dense layers
    layers.append(keras.layers.Flatten(name='flatten'))

    # Build dense layers with dropout regularization
    neurons = starting_dense_neurons
    
    for i in range(num_dense_layers):
        layers.append(
            keras.layers.Dense(
                neurons, 
                activation='relu',
                name=f'dense_{i+1}'
            )
        )
        layers.append(
            keras.layers.Dropout(
                dropout_rate,
                name=f'dropout_{i+1}'
            )
        )
        # Double neurons for next layer
        neurons *= 2

    # Binary classification output layer
    layers.append(
        keras.layers.Dense(
            1, 
            activation='sigmoid',
            name='output'
        )
    )

    # Create model
    model = keras.Sequential(layers, name='STM_Tip_Classifier')
    
    # Compile model with binary classification settings
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_thesis_architecture(surface='si111_7x7', input_shape=(700, 700, 1)):
    """
    Returns the exact CNN architecture used in the PhD thesis for a given surface.
    
    These architectures were determined through extensive hyperparameter search
    and validated on experimental data.
    
    Args:
        surface (str): Surface type. Options:
            - 'si111_7x7': Silicon (111) 7×7 reconstruction
            - 'bsi': Boron-doped silicon
            
        input_shape (tuple): Shape of input images.
            Default: (700, 700, 1) for full-resolution thesis models
    
    Returns:
        tf.keras.Model: Pre-configured model matching thesis architecture
    
    Example:
        >>> model = get_thesis_architecture('si111_7x7')
        >>> model.fit(train_data, epochs=100)
    """
    if surface == 'si111_7x7':
        # Si(111)-7×7 optimal architecture
        # Conv filters: [20, 40, 60, 80, 100]
        # Dense neurons: [32, 64, 128]
        # Achieved 96% accuracy
        return build_model(
            num_conv_layers=5,
            starting_conv_filters=20,
            filter_multiplier=2,
            num_dense_layers=3,
            starting_dense_neurons=32,
            dropout_rate=0.5,
            input_shape=input_shape
        )
    
    elif surface == 'bsi':
        # B:Si optimal architecture (same as Si(111)-7×7)
        # Achieved 90% accuracy, 97% precision
        return build_model(
            num_conv_layers=5,
            starting_conv_filters=20,
            filter_multiplier=2,
            num_dense_layers=3,
            starting_dense_neurons=32,
            dropout_rate=0.5,
            input_shape=input_shape
        )
    
    else:
        raise ValueError(
            f"Unknown surface type: {surface}. "
            f"Supported types: 'si111_7x7', 'bsi'"
        )


# Model configuration presets for quick experimentation
MODEL_PRESETS = {
    'small': {
        'num_conv_layers': 3,
        'starting_conv_filters': 16,
        'filter_multiplier': 2,
        'num_dense_layers': 1,
        'starting_dense_neurons': 32,
        'description': 'Lightweight model for quick experiments'
    },
    'medium': {
        'num_conv_layers': 4,
        'starting_conv_filters': 16,
        'filter_multiplier': 2,
        'num_dense_layers': 2,
        'starting_dense_neurons': 64,
        'description': 'Balanced model with good performance/speed tradeoff'
    },
    'large': {
        'num_conv_layers': 5,
        'starting_conv_filters': 20,
        'filter_multiplier': 2,
        'num_dense_layers': 3,
        'starting_dense_neurons': 32,
        'description': 'Full thesis architecture for best accuracy'
    }
}


def build_preset_model(preset='medium', input_shape=(140, 140, 1)):
    """
    Build a model from predefined configuration presets.
    
    Args:
        preset (str): Preset name ('small', 'medium', 'large')
        input_shape (tuple): Input image shape
        
    Returns:
        tf.keras.Model: Configured model
        
    Example:
        >>> model = build_preset_model('large')
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available presets: {list(MODEL_PRESETS.keys())}"
        )
    
    config = MODEL_PRESETS[preset].copy()
    config.pop('description')  # Remove description from kwargs
    
    return build_model(**config, input_shape=input_shape)
