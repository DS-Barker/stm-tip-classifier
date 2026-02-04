"""
Model Evaluation for STM Tip Classifier
========================================

Evaluates a trained CNN model on a test dataset and reports performance metrics.

Usage:
    from src.cnn_classifier.evaluate import evaluate_model
    
    # Evaluate a saved model
    results = evaluate_model(
        model_path='models/si111_final.h5',
        test_data_dir='data/si111/processed/test',
        target_size=(700, 700)
    )
    
    print(f"Test accuracy: {results['accuracy']:.2%}")
    print(f"Test loss: {results['loss']:.4f}")

Author: Dylan S. Barker, University of Leeds, 2024
"""

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from pathlib import Path

import os
import time


def convert_to_preferred_format(sec):
    """Convert seconds to HH:MM:SS format."""
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def evaluate_model(model_path, test_data_dir, target_size=(700, 700), batch_size=16):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path (str or Path): Path to saved model (.h5 file)
        test_data_dir (str or Path): Directory containing test data with Good/Bad subdirectories
        target_size (tuple): Image dimensions (height, width). Default: (700, 700)
        batch_size (int): Batch size for evaluation. Default: 16
    
    Returns:
        dict: Evaluation results containing:
            - 'loss': Test loss
            - 'accuracy': Test accuracy
            - 'evaluation_time': Time taken for evaluation
            - 'class_names': Mapping of class indices to names
    
    Example:
        >>> results = evaluate_model(
        ...     model_path='models/my_model.h5',
        ...     test_data_dir='data/bsi/processed/test'
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    # Find the start time
    initial_t = time.perf_counter()
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    model.summary()
    
    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    test_datagen = ImageDataGenerator(rescale=1./255.0)
    
    # Pass in the appropriate arguments to the flow_from_directory method
    test_generator = test_datagen.flow_from_directory(directory=test_data_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=target_size,
                                                        color_mode='grayscale')
    
    class_names = test_generator.class_indices
    class_names = {y: x for x, y in class_names.items()}
    
    scores = model.evaluate(test_generator)
    
    # Find the final time (before plotting etc.)
    final_t = time.perf_counter()
    
    total_time = final_t - initial_t
    print('*'*20)
    print(f'Total time taken for evaluation was {convert_to_preferred_format(total_time)}')
    print('*'*20)
    
    # Return results as dictionary
    return {
        'loss': scores[0],
        'accuracy': scores[1],
        'evaluation_time': total_time,
        'class_names': class_names
    }


# If running as script (not imported)
if __name__ == "__main__":

    # Change later to fit with Git structure
    DATA_DIR = Path("INSERT TEST DATA DIRECTORY HERE")
    MODEL_PATH = Path("INSERT PATH TO .h5 MODEL FILE HERRE")
    
    results = evaluate_model(
        model_path=MODEL_PATH,
        test_data_dir=DATA_DIR,
        target_size=(700, 700),
        batch_size=16
    )
    
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Class mapping: {results['class_names']}")