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
- Model checkpoint saving
- Training history export to CSV
- Automatic results organization

Usage:
    from src.cnn_classifier.train import train_single_model
    from src.cnn_classifier.model import build_model
    from src.data_processing.generators import train_val_generators
    
    # Setup data
    train_gen, val_gen = train_val_generators(
        training_dir='data/bsi/processed/training',
        validation_dir='data/bsi/processed/validation'
    )
    
    # Build model
    model = build_model(
        num_conv_layers=5,
        starting_conv_filters=20,
        num_dense_layers=3,
        starting_dense_neurons=32
    )
    
    # Train it
    trained_model, history = train_single_model(
        model=model,
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=100,
        output_dir='results/my_experiment',
        model_name='bsi_final'
    )
    
    # Access results
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_acc:.2%}")

Outputs:
    results/my_experiment/
    ├── bsi_final.h5              # Saved model
    ├── bsi_final_training.png    # Accuracy/loss plots
    └── bsi_final_history.csv     # Training metrics per epoch

Author: Dylan S. Barker, University of Leeds, 2024
"""

def train_single_model(model, 
                      train_generator, 
                      validation_generator,
                      epochs=30,
                      output_dir=None,
                      save_model=True,
                      save_plots=True,
                      save_history=True,
                      model_name='model',
                      verbose=1):
    """
    Train a single CNN model and optionally save results.
    
    Args:
        model (tf.keras.Model): Compiled Keras model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs (int): Number of training epochs. Default: 30
        output_dir (str or Path): Directory to save results. If None, results not saved
        save_model (bool): Whether to save trained model. Default: True
        save_plots (bool): Whether to save accuracy/loss plots. Default: True
        save_history (bool): Whether to save training history CSV. Default: True
        model_name (str): Base name for saved files. Default: 'model'
        verbose (int): Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
            Default: 1
    
    Returns:
        tuple: (model, history)
            - model: Trained Keras model
            - history: Training history object containing metrics
    
    Example:
        >>> from src.cnn_classifier.model import build_model
        >>> from src.data_processing.generators import train_val_generators
        >>> 
        >>> # Setup
        >>> model = build_model(num_conv_layers=5, ...)
        >>> train_gen, val_gen = train_val_generators(...)
        >>> 
        >>> # Train
        >>> trained_model, history = train_single_model(
        ...     model=model,
        ...     train_generator=train_gen,
        ...     validation_generator=val_gen,
        ...     epochs=100,
        ...     output_dir='results/si111_experiment',
        ...     model_name='si111_final'
        ... )
        >>> 
        >>> # Access metrics
        >>> final_acc = history.history['val_accuracy'][-1]
        >>> print(f"Final validation accuracy: {final_acc:.2%}")
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Train the model
    print(f"\nTraining model for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_generator
    )
    
    # Print final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nTraining complete!")
    print(f"Final train accuracy: {final_train_acc:.4f}, loss: {final_train_loss:.4f}")
    print(f"Final val accuracy: {final_val_acc:.4f}, loss: {final_val_loss:.4f}")
    
    # Save results if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trained model
        if save_model:
            model_path = output_dir / f'{model_name}.h5'
            model.save(model_path)
            print(f"Model saved to: {model_path}")
        
        # Save training plots
        if save_plots:
            _save_training_plots(history, output_dir, model_name)
        
        # Save training history
        if save_history:
            _save_training_history(history, output_dir, model_name)
    
    return model, history


def _save_training_plots(history, output_dir, model_name):
    """
    Create and save accuracy and loss plots.
    
    Args:
        history: Keras training history
        output_dir (Path): Directory to save plots
        model_name (str): Base name for plot file
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(epochs_range, acc, 'r', label='Training accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, 'b', label='Validation accuracy', linewidth=2)
    ax1.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(epochs_range, loss, 'r', label='Training loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, 'b', label='Validation loss', linewidth=2)
    ax2.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'{model_name}_training.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to: {plot_path}")


def _save_training_history(history, output_dir, model_name):
    """
    Save training history as CSV file.
    
    Args:
        history: Keras training history
        output_dir (Path): Directory to save CSV
        model_name (str): Base name for CSV file
    """
    import pandas as pd
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # Convert history to DataFrame
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = range(1, len(hist_df) + 1)
    
    # Reorder columns to put epoch first
    cols = ['epoch'] + [col for col in hist_df.columns if col != 'epoch']
    hist_df = hist_df[cols]
    
    # Save to CSV
    csv_path = output_dir / f'{model_name}_history.csv'
    hist_df.to_csv(csv_path, index=False)
    
    print(f"Training history saved to: {csv_path}")