import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import  Dense
from functions import *
import shutil
# Create Keras model
# Returns the Keras model
def createModel(fit=False):
    # Downloading from network VGG16 Keras model
    vgg16_model = tf.keras.applications.vgg16.VGG16()
    # Creating our Sequential model
    model=Sequential()
    for layer in vgg16_model.layers[:-1]:
        # Adding all layers, that VGG16 model already have
        model.add(layer)
    for layer in model.layers:
        # Setting them to non_trainable
        layer.trainable= False

    # Adding 2 units to our Model, with ship and without ship
    model.add(Dense(units=2,activation='softmax'))

    # If uncomment this row we coould this what layers our model have
    # model.summary()

    # Compaling model
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    # Training our model or loading her, depends on given parameters 
    if(fit):
        fitModel(model)
    else:
        model=load_model(os.getcwd()+'/model_saves/seqModel12.h5')
    return model


# Train the model
# Trained model saves in directory 'model_saves'
def fitModel(model):
    # Getting training data by using function 'prepareData'
    train_batches,valid_batches=prepareData()
    # Giving traing_batches, these are our 'train_images', and valid_bathes,these are 'validation_images'
    model.fit(x=train_batches,validation_data=valid_batches,epochs=60,verbose=2)
    # Saving model to directory
    model.save(os.getcwd()+'/model_saves/seqModel12.h5')


# Function that gives predicts from our model and copy given file to 'models_predictions' folder, depends on image
# Returns the results of predictions or None
def predictModel(model,image_paths):
    # predictions=[]
    for image_file in image_paths:
        # Creating from path files to format which receives Keras model
        img_path = image_file
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        # Giving to func array of images, verbose parameter responsible for show results of each epoch, 0 means nothing to show
        predictions=model.predict(x=img_array,verbose=0)
        #  If the first value of 'prediction_array' is lesser thet second,
        # it means that model thinks that this is image without ship and copy this file to folder without  
        if predictions[0][0]>predictions[0][1]:
            print("Image withouth ship")
            shutil.copy(img_path,os.getcwd()+'/models_predictions/without')
        else:
            print("Image with ship")
            shutil.copy(img_path, os.getcwd()+'/models_predictions/with')
    try:
        predictions.any()
    except NameError:
        print("Any images were found")
        return None
    return predictions
