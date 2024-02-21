from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import confusion_matrix

# Prepare the default data for model
# Returns train_batches,valid and test
def prepareData(with_tests=False):
    # Setting folders for our model such as train,valid and test
    train_path = os.getcwd()+'/data/train'
    valid_path = os.getcwd()+'/data/valid'
    test_path = os.getcwd()+'/data/test'
    # Converting them to format which receives the Keras model
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(224,224), classes=['not_ships', 'ships'], batch_size=2)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['not_ships', 'ships'], batch_size=2)
    if(with_tests):
        test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=test_path, target_size=(224,224), classes=['not_ships', 'ships'], batch_size=2, shuffle=False)

        return train_batches,valid_batches,test_batches
    
    return train_batches,valid_batches


# Function returns default test files for model
def findTestImages():
    # Setting default path to files
    path=os.getcwd()+'/data/test/'

    image_paths = []
    # My test images named 'in81.jpg' to 'in137jpg' and 'not81' to 'not137',
    # so that we can easily see what images our model has put correctly and what not.
    #This filenames I append to array 'image_paths' 
    for i in range(81,138):
        image_paths.append(path+'in'+str(i)+'.jpg')
    for i in range(81,138):
        if(i>=101):
            i+=300
        image_paths.append(path+'not'+str(i)+'.jpg')
    return image_paths


# Cleans directory with 'models_predictions'
def clearModelsPredictionsDirectory():
    clearDirectory(os.getcwd()+'/models_predictions/with/*')
    clearDirectory(os.getcwd()+'/models_predictions/without/*')

# Cleans directory depends on given path
def clearDirectory(path_to_directory):
    files = glob.glob(path_to_directory)
    for f in files:
        os.remove(f)

# Find images_paths_array for Keras model
# Returns array of image's paths
def findGivenImagesArray():
    images_paths = []
    # Finding all files in 'your_images' directory
    files = glob.glob(os.getcwd()+'/your_images/*')
    # Appending them to our 'images_paths_array'
    for f in files:
        images_paths.append(f)
    return images_paths
