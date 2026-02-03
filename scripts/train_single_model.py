"""
Training Utilities for STM Tip Classifier
==========================================

Provides training functions for CNN models, including single model training,
evaluation, and results visualization.

This module separates training logic from model architecture and hyperparameter
search, making it easy to train individual models with specific configurations.

Features:
- Single model training with configurable parameters
- Training history visualization (accuracy and loss curves)
- Model evaluation with metrics calculation
- Results saving (models, plots, histories)

Usage:
    from src.cnn_classifier.train import train_single_model
    from src.cnn_classifier.model import build_model
    from src.data_processing.generators import train_val_generators
    
    # Setup data
    train_gen, val_gen = train_val_generators(...)
    
    # Build model
    model = build_model(
        num_conv_layers=5,
        starting_conv_filters=20,
        num_dense_layers=3,
        starting_dense_neurons=32
    )
    
    # Train it
    model, history = train_single_model(
        model=model,
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=100,
        output_dir='results/my_experiment'
    )

Author: Dylan S. Barker, University of Leeds, 2024
"""

import os
from pathlib import Path
import argparse
from shutil import copyfile
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# Import from src package
from src.cnn_classifier.model import build_model
from src.data_processing.generators import train_val_generators


def create_train_val_dirs(root_path):
    """
    Creates directories for the train and validation sets.
    
    Args:
        root_path (Path or str): Base directory path to create subdirectories from
    
    Returns:
        None
    """ 
    root_path = Path(root_path)
    root_path.mkdir(parents=True, exist_ok=True)

    val_path = root_path / 'validation'
    train_path = root_path / 'training'
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    (val_path / 'Good').mkdir(exist_ok=True)
    (val_path / 'Bad').mkdir(exist_ok=True)
    (train_path / 'Good').mkdir(exist_ok=True)
    (train_path / 'Bad').mkdir(exist_ok=True)


def split_data(source_dir, training_dir, validation_dir, split_size):
    """
    Splits the data into train and validation sets.
    
    Args:
        source_dir (Path): Directory path containing the images
        training_dir (Path): Directory path to be used for training
        validation_dir (Path): Directory path to be used for validation
        split_size (float): Proportion of the dataset to be used for training
        
    Returns:
        None
    """
    source_dir = Path(source_dir)
    training_dir = Path(training_dir)
    validation_dir = Path(validation_dir)
    
    im_list = random.sample(list(source_dir.iterdir()), len(list(source_dir.iterdir())))

    for index, im_path in enumerate(im_list):
        if im_path.stat().st_size == 0:
            print(f"{im_path.name} is zero length, so ignoring.")
            continue
            
        if index <= split_size * len(im_list):
            copyfile(im_path, training_dir / im_path.name)
        else:
            copyfile(im_path, validation_dir / im_path.name)


def clear_directory(directory):
    """
    Remove all files in a directory.
    
    Args:
        directory (Path): Directory to clear
    """
    directory = Path(directory)
    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                file.unlink()


def save_results(model, history, output_dir, save_name, args):
    """
    Save model, training history, and accuracy plots.
    
    Args:
        model: Trained Keras model
        history: Training history object
        output_dir (Path): Directory to save results
        save_name (str): Base name for saved files
        args: Command line arguments (for metadata)
    """
    output_dir = Path(output_dir)
    
    # Save model
    model_path = output_dir / 'models' / f'{save_name}.h5'
    model.save(model_path)
    print(f"  Saved model to: {model_path}")
    
    # Extract metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = range(len(acc))
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, acc, 'r', label='Training accuracy')
    plt.plot(epoch_range, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, loss, 'r', label='Training loss')
    plt.plot(epoch_range, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'plots' / f'{save_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to: {plot_path}")
    
    # Save history as CSV with metadata
    hist_df = pd.DataFrame(history.history)
    
    hist_path = output_dir / 'histories' / f'{save_name}.csv'
    hist_df.to_csv(hist_path, index=False)
    print(f"  Saved history to: {hist_path}")
    
    # Print final metrics
    final_acc = acc[-1]
    final_val_acc = val_acc[-1]
    final_loss = loss[-1]
    final_val_loss = val_loss[-1]
    print(f"  Final train accuracy: {final_acc:.4f}, loss: {final_loss:.4f}")
    print(f"  Final val accuracy: {final_val_acc:.4f}, loss: {final_val_loss:.4f}")


def main(args):
    """
    Model training.
    
    Args:
        args: Command line arguments containing model architecture
    """
    # Set environment variables for TensorFlow
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Setup paths relative to project root
    project_root = Path(args.project_root)
    data_dir = project_root / 'data' / args.surface
    source_dir = data_dir / 'raw'
    work_dir = data_dir / 'processed'
    output_dir = project_root / 'results' / args.surface / 'output_models'
    
    # Create output directories
    s(output_dir / 'models').mkdir(parents=True, exist_ok=True)
    s(output_dir / 'histories').mkdir(parents=True, exist_ok=True)
    s(output_dir / 'plots').mkdir(parents=True, exist_ok=True)

    # Setup data directories
    good_source = source_dir / 'good'
    bad_source = source_dir / 'bad'
    
    training_dir = work_dir / 'training'
    validation_dir = work_dir / 'validation'
    
    training_good = training_dir / 'Good'
    training_bad = training_dir / 'Bad'
    validation_good = validation_dir / 'Good'
    validation_bad = validation_dir / 'Bad'

    # Create directory structure
    try:
        create_train_val_dirs(work_dir)
        print(f'Created directory structure at {work_dir}')
    except FileExistsError:
        print(f"{work_dir} already exists.")

    # Clear existing data
    print("Clearing existing processed data...")
    for directory in [training_good, training_bad, validation_good, validation_bad]:
        clear_directory(directory)

    # Split data into train/val
    print(f"Splitting data with {args.split_size:.1%} for training...")
    split_data(good_source, training_good, validation_good, args.split_size)
    split_data(bad_source, training_bad, validation_bad, args.split_size)

    # Create data generators using imported function
    print("Creating data generators...")
    train_generator, validation_generator = train_val_generators(
        training_dir, 
        validation_dir, 
        batch_size=args.batch_size, 
        target_size=args.target_size
    )

      
    # Begin training
    
                    
    print(f"\n{'='*70}")
    print(f"Building model")
    print(f"{'='*70}")
    
    # Build model using imported function
    model = build_model(
        num_conv_layers=args.conv_layers,
        starting_conv_filters=args.conv_filters,
        filter_multiplier=args.filter_multiplier,
        num_dense_layers=args.dense_layers,
        starting_dense_neurons=args.dense_neurons,
        conv_kernel_size=tuple(args.conv_kernel_size),
        pool_size=tuple(args.pool_size),
        dropout_rate=args.dropout_rate,
        input_shape=(*args.target_size, 1)
    )
    
    if args.verbose:
        print("\nModel Architecture:")
        model.summary()
    
    # Train model
    print("\nTraining...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=1 if args.verbose else 2,
        validation_data=validation_generator
    )
    
    # Create descriptive save name
    save_name = (f'{args.epochs}epochs_'
                f'conv{args.conv_layers}x{args.conv_filters}_'
                f'dense{args.dense_layers}x{args.dense_neurons}')
    
    # Save results
    save_results(model, history, output_dir, save_name, args)
    
    # Add to summary
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
        
    # Clear session to free memory
    tf.keras.backend.clear_session()
        
    print("\n" + "="*70)
    print("Training!")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Single model train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    parser.add_argument(
        '--project_root',
        type=str,
        default='.',
        help='Root directory of the project'
    )
    parser.add_argument(
        '--surface',
        type=str,
        default='bsi',
        choices=['si111_7x7', 'bsi', 'cu111_c60', 'cu111_co'],
        help='Surface type being classified'
    )
    
    # Hyperparameters to search - Convolutional Layers
    parser.add_argument(
        '--conv_layers',
        type=int,
        nargs='+',
        default=3,
        help='Number of convolutional layers to test'
    )
    parser.add_argument(
        '--conv_filters',
        type=int,
        nargs='+',
        default=16,
        help='Starting filter counts for conv layers'
    )
    parser.add_argument(
        '--filter_multiplier',
        type=int,
        default=2,
        help='Multiplier for filters in successive conv layers'
    )
    parser.add_argument(
        '--conv_kernel_size',
        type=int,
        nargs=2,
        default=[3, 3],
        help='Kernel size for convolutional layers'
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        nargs=2,
        default=[2, 2],
        help='Pool size for max pooling layers'
    )
    
    # Hyperparameters to search - Dense Layers
    parser.add_argument(
        '--dense_neurons',
        type=int,
        nargs='+',
        default=32,
        help='Starting neuron counts for dense layers'
    )
    parser.add_argument(
        '--dense_layers',
        type=int,
        nargs='+',
        default=3,
        help='Number of dense layers'
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='Dropout rate for dense layers'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[140, 140],
        help='Target image size (height width)'
    )
    parser.add_argument(
        '--split_size',
        type=float,
        default=0.9,
        help='Proportion of data for training (vs validation)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed training output and model summaries'
    )
    
    args = parser.parse_args()
    args.target_size = tuple(args.target_size)
    
    main(args)
