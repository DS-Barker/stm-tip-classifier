from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import random
from shutil import copyfile

def create_train_val_dirs(root_path):
    """
    Creates directories for the train and test sets
    
    Args:
        root_path (string) - the base directory path to create subdirectories from
    
    Returns:
        None
    """ 

    os.makedirs(root_path)

    val_path = os.path.join(root_path, 'validation')
    train_path = os.path.join(root_path, 'training')
    os.makedirs(train_path)
    os.makedirs(val_path)

    os.makedirs(os.path.join(val_path, 'Good'))
    os.makedirs(os.path.join(val_path, 'Bad'))
    os.makedirs(os.path.join(train_path, 'Good'))
    os.makedirs(os.path.join(train_path, 'Bad'))

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):

    """
    Splits the data into train and test sets
    
    Args:
        SOURCE_DIR (string): directory path containing the images
        TRAINING_DIR (string): directory path to be used for training
        VALIDATION_DIR (string): directory path to be used for validation
        SPLIT_SIZE (float): proportion of the dataset to be used for training
        
    Returns:
        None
    """
    

    im_list = random.sample(os.listdir(SOURCE_DIR), len(os.listdir(SOURCE_DIR)))

    for index, im in enumerate(im_list):
        if os.path.getsize(os.path.join(SOURCE_DIR, im)) == 0:
            print(f"{im} is zero length, so ignoring.")
            continue
        if index <= SPLIT_SIZE*len(im_list):
            copyfile(os.path.join(SOURCE_DIR, im), os.path.join(TRAINING_DIR, im))
        else:
            copyfile(os.path.join(SOURCE_DIR, im), os.path.join(VALIDATION_DIR, im))

def train_val_generators(TRAINING_DIR, VALIDATION_DIR, batch_size, input_size):
    """
    Creates the training and validation data generators
  
    Args:
        TRAINING_DIR (string): directory path containing the training images
        VALIDATION_DIR (string): directory path containing the testing/validation images
        batch_size (integer): Integer number of batches (amount of images trained per loop).
        input_size (tensor: (int, int)): Tensor containing the size of the input image, or 
        preferred input size (if different to the actual size it will be scaled). 
    
    Returns:
        train_generator, validation_generator - tuple containing the generators
    """

    # Instantiate the ImageDataGenerator class (don't forget to set the arguments to augment the images)
    train_datagen = ImageDataGenerator(rescale=1./255.0,
                                    #  rotation_range=40,
                                    #  width_shift_range=0.2,
                                    #  height_shift_range=0.2,
                                    #  shear_range=0.2,
                                    #  zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    #  fill_mode='nearest'
                                    )

    # Pass in the appropriate arguments to the flow_from_directory method
    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=target_size,
                                                        color_mode= 'grayscale')

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    validation_datagen = ImageDataGenerator(rescale=1./255.0)

    # Pass in the appropriate arguments to the flow_from_directory method
    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                target_size=target_size,
                                                                color_mode= 'grayscale')
    ### END CODE HERE
    return train_generator, validation_generator

# set tf_xla_enable_xla_devices flags
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define paths
SSD_DIR = Path(os.getcwd())
DATA_DIR = SSD_DIR.joinpath(r'Python_scripts\Machine_Learning\Si111\Tensorflow\Dataset\Binary_notc\all\Splitting')
SOURCE_DIR = DATA_DIR.joinpath('Input_data')
ROOT_DIR = DATA_DIR.joinpath('ForRuns')

GOOD_SOURCE_DIR = os.path.join(SOURCE_DIR, "Good")
BAD_SOURCE_DIR = os.path.join(SOURCE_DIR,"Bad")

TRAINING_DIR = os.path.join(ROOT_DIR, 'training')
VALIDATION_DIR = os.path.join(ROOT_DIR, 'validation')

TRAINING_GOOD_DIR = os.path.join(TRAINING_DIR, "Good/")
VALIDATION_GOOD_DIR = os.path.join(VALIDATION_DIR, "Good/")

TRAINING_BAD_DIR = os.path.join(TRAINING_DIR, "Bad/")
VALIDATION_BAD_DIR = os.path.join(VALIDATION_DIR, "Bad/")

try:
  create_train_val_dirs(root_path=ROOT_DIR)
  print(f'Creating directory at {ROOT_DIR}...')
except FileExistsError:
  print(f"{ROOT_DIR} already exists.")

# Empty directories in case you run this cell multiple times
if len(os.listdir(TRAINING_BAD_DIR)) > 0:
    for file in os.scandir(TRAINING_BAD_DIR):
        os.remove(file.path)
if len(os.listdir(TRAINING_GOOD_DIR)) > 0:
    for file in os.scandir(TRAINING_GOOD_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_BAD_DIR)) > 0:
    for file in os.scandir(VALIDATION_BAD_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_GOOD_DIR)) > 0:
    for file in os.scandir(VALIDATION_GOOD_DIR):
        os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# Run the function
# NOTE: Messages about zero length images should be printed out
split_data(GOOD_SOURCE_DIR, TRAINING_GOOD_DIR, VALIDATION_GOOD_DIR, split_size)
split_data(BAD_SOURCE_DIR, TRAINING_BAD_DIR, VALIDATION_BAD_DIR, split_size)

batch_size = 32
target_size = (140, 140)
epochs = 30

train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR, batch_size=batch_size, input_size=target_size)

neurons = [32,64,128,256,512,1024] 
# Apparently a maximum of two layers is all that we would ever need - I'll look into the theory later. 
num_layers = [1,2]

for i in neurons:
  for j in num_layers:
    tmp_i = i
    # Set initial layers - Usually input, conv and flatten before dense layers
    layers = [
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape= (140,140,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten()]

    for k in range(1, j+1):
      layers.append(tf.keras.layers.Dense(tmp_i, activation='relu'))
      layers.append(tf.keras.layers.Dropout(0.5))
      tmp_i *= 2

    # Add in the output layer
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    

# model = tf.keras.models.Sequential([ 
      
#       layers.Conv2D(16, (3,3), activation='relu', input_shape= (140,140,1)),
#       layers.MaxPooling2D((2,2)),
#       layers.Conv2D(32, (3,3), activation='relu'),
#       layers.MaxPooling2D((2,2)),
#       layers.Conv2D(64, (3,3), activation='relu'),
#       layers.MaxPooling2D((2,2)),
#       layers.Conv2D(128, (3,3), activation='relu'),
#       layers.MaxPooling2D((2,2)),
#       layers.Flatten(),
#       layers.Dense(256, activation='relu'),
#       layers.Dropout(0.5),
#       layers.Dense(512, activation='relu'),
#       layers.Dropout(0.5),
#       layers.Dense(1024, activation='relu'),
#       layers.Dropout(0.5),
#       layers.Dense(1, activation='sigmoid'),
#   ])

    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Bigin training!
    history = model.fit(train_generator,
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator)


    # Save the model, history and accuracy plot
    save_name = f'{epochs}epochs_{j}layers_{i}Starting'
    Models_dir = SSD_DIR.joinpath(r'ML\Models\TensorFlow\Si(111)')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch_range = range(len(acc))

    plt.plot(epoch_range, acc, 'r', label='Training accuracy')
    plt.plot(epoch_range, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)

    plt.savefig(Models_dir.joinpath(f'acc_plots/HT/{save_name}.png'))
    model.save(Models_dir.joinpath(f'models/{save_name}.h5'))


    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # or save to csv: 
    hist_csv_file = Models_dir.joinpath(f'histories/{save_name}.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    tf.keras.backend.clear_session()
    plt.close()