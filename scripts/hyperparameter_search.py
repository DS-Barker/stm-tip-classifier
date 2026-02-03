"""
Hyperparameter Search for CNN Tip Classifier
============================================

Grid search over CNN architectures to find optimal configuration for
STM tip state classification.

Tests combinations of:
- Convolutional layer depth and filter counts
- Dense layer depth and neuron counts
- Saves all models, histories, and performance plots

Usage:
    # Quick test
    python scripts/hyperparameter_search.py \
        --conv_layers 3 4 \
        --dense_layers 1 2 \
        --epochs 10
    
    # Full search
    python scripts/hyperparameter_search.py \
        --conv_layers 3 4 5 \
        --conv_filters 16 20 32 \
        --dense_neurons 32 64 128 \
        --dense_layers 1 2 3 \
        --epochs 30

Output: results/surface/hyperparameter_search/
    ├── models/          # Saved .h5 files
    ├── histories/       # Training histories
    ├── plots/           # Accuracy/loss curves
    └── search_summary.csv  # All runs ranked by performance

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
    Main hyperparameter search loop.
    
    Args:
        args: Command line arguments containing hyperparameter ranges
    """
    # Set environment variables for TensorFlow
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Setup paths relative to project root
    project_root = Path(args.project_root)
    data_dir = project_root / 'data' / args.surface
    source_dir = data_dir / 'raw'
    work_dir = data_dir / 'processed'
    output_dir = project_root / 'results' / args.surface / 'hyperparameter_search'
    
    # Create output directories
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'histories').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)

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

    # Hyperparameter grid search
    print("\n" + "="*70)
    print("Starting Hyperparameter Search")
    print("="*70)
    
    # Calculate total runs
    total_runs = (len(args.conv_layers) * len(args.conv_filters) * 
                  len(args.dense_neurons) * len(args.dense_layers))
    current_run = 0
    
    # Create summary file
    summary_path = output_dir / 'search_summary.csv'
    summary_data = []
    
    # Search over convolutional configurations
    for num_conv in args.conv_layers:
        for starting_conv_filters in args.conv_filters:
            # Calculate actual filter progression
            filter_list = [starting_conv_filters * (args.filter_multiplier ** i) 
                           for i in range(num_conv)]
            
            # Search over dense layer configurations
            for starting_dense_neurons in args.dense_neurons:
                for num_dense_layers in args.dense_layers:
                    current_run += 1
                    
                    print(f"\n{'='*70}")
                    print(f"Run {current_run}/{total_runs}")
                    print(f"{'='*70}")
                    print(f"Conv layers: {num_conv} with filters {filter_list}")
                    print(f"Dense layers: {num_dense_layers}, starting neurons: {starting_dense_neurons}")
                    
                    # Build model using imported function
                    model = build_model(
                        num_conv_layers=num_conv,
                        starting_conv_filters=starting_conv_filters,
                        filter_multiplier=args.filter_multiplier,
                        num_dense_layers=num_dense_layers,
                        starting_dense_neurons=starting_dense_neurons,
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
                                f'conv{num_conv}x{starting_conv_filters}_'
                                f'dense{num_dense_layers}x{starting_dense_neurons}')
                    
                    # Save results
                    save_results(model, history, output_dir, save_name, args)
                    
                    # Add to summary
                    final_acc = history.history['accuracy'][-1]
                    final_val_acc = history.history['val_accuracy'][-1]
                    summary_data.append({
                        'run': current_run,
                        'num_conv_layers': num_conv,
                        'conv_filters': str(filter_list),
                        'num_dense_layers': num_dense_layers,
                        'starting_dense_neurons': starting_dense_neurons,
                        'final_train_acc': final_acc,
                        'final_val_acc': final_val_acc,
                        'model_file': f'{save_name}.h5'
                    })
                    
                    # Clear session to free memory
                    tf.keras.backend.clear_session()
    
    # Save summary of all runs
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('final_val_acc', ascending=False)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*70)
    print("Hyperparameter Search Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"\nTop 5 models by validation accuracy:")
    print(summary_df[['run', 'num_conv_layers', 'num_dense_layers', 
                      'final_val_acc']].head())
    print(f"\nFull results in: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for STM tip classifier CNN',
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
        default=[3, 4, 5],
        help='Number of convolutional layers to test'
    )
    parser.add_argument(
        '--conv_filters',
        type=int,
        nargs='+',
        default=[16, 20, 32],
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
        default=[32, 64, 128, 256],
        help='Starting neuron counts for dense layers'
    )
    parser.add_argument(
        '--dense_layers',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help='Number of dense layers to test'
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
