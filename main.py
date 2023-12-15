#Importing required libraries

from flask import Flask, render_template, request, redirect, url_for

import os
import gc

from keras.models import load_model

import numpy as np 
import pandas as pd 

from keras import Sequential
from keras.callbacks import EarlyStopping

from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from keras.utils import to_categorical
# from keras import regularizers
from tensorflow.keras.models import Model


from keras.preprocessing.image import load_img
#Function to read images from test directory



    
#function to extract features from the dataset by a given pretrained model
img_size = (331,331,3)
model = load_model('IA_model/dog_breed_classifier.h5')

labels = pd.read_csv('IA_model/labels.csv')

classes = sorted(list(set(labels['breed'])))

# def images_to_array_test(test_path, img_size = (331,331,3)):
#     test_filenames = [test_path + fname for fname in os.listdir(test_path)]

#     data_size = len(test_filenames)
#     images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    
    
#     for ix,img_dir in enumerate(tqdm(test_filenames)):
# #         img_dir = os.path.join(directory, image_name + '.jpg')
#         img = load_img(img_dir, target_size = img_size)
# #         img = np.expand_dims(img, axis=0)
# #         img = processed_image_resnet(img)
# #         img = img/255
#         images[ix]=img
# #         images[ix] = img_to_array(img)
#         del img
#     print('Ouptut Data Size: ', images.shape)
#     return images

# test_data = images_to_array_test('dog-breed-identification/test/', img_size)

def get_features(model_name, model_preprocessor, input_size, data):

    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    #Extract feature.
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps



def extact_features(data):

    # Extract features using InceptionV3 
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    inception_preprocessor = preprocess_input
    inception_features = get_features(InceptionV3,
                                      inception_preprocessor,
                                      img_size, data)
    
    # Extract features using Xception 
    from keras.applications.xception import Xception, preprocess_input
    xception_preprocessor = preprocess_input
    xception_features = get_features(Xception,
                                     xception_preprocessor,
                                     img_size, data)
    
    # Extract features using InceptionResNetV2 
    from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
    inc_resnet_preprocessor = preprocess_input
    inc_resnet_features = get_features(InceptionResNetV2,
                                       inc_resnet_preprocessor,
                                       img_size, data)
    inception_features,xception_features,inc_resnet_features

    # Extract features using NASNetLarge 
    from keras.applications.nasnet import NASNetLarge, preprocess_input
    nasnet_preprocessor = preprocess_input
    nasnet_features = get_features(NASNetLarge,
                                   nasnet_preprocessor,
                                   img_size, data)     


    final_features = np.concatenate([inception_features,
                                     xception_features,
                                     nasnet_features,
                                     inc_resnet_features],axis=-1)
    
    print('Final feature maps shape', final_features.shape)
    
    #deleting to free up ram memory
    del inception_features
    del xception_features
    del nasnet_features
    del inc_resnet_features
    gc.collect()
    
    
    return final_features

def rename_jpg_files(directory_path,newName):
    # Change the current working directory to the specified path
    os.chdir(directory_path)

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a JPG file
        if filename.lower().endswith(".jpg"):
            # Construct the new name (you can modify this based on your requirements)
            new_name = "new_" + filename

            # Rename the file
            os.rename(filename, newName)
            print(f"Renamed: {filename} to {new_name}")
            
            
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
     
    file_name = request.args.get('file')
    name = request.args.get('name')
    pre=request.args.get('pre')
    return render_template('home.html', file_name=file_name,name=name,pre=pre)

    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(file_path)
    #reading the image and converting it into an np array
    #'dog-breed-identification/train/00214f311d5d2247d5dfe4fe24b2303d.jpg'
    img_g = load_img(file_path,target_size = img_size)
    img_g = np.expand_dims(img_g, axis=0) # as we trained our model in (row, img_height, img_width, img_rgb) format, np.expand_dims convert the image into this format
    # img_g

    # #Predict test labels given test data features.
    test_features = extact_features(img_g)

    #test_features = extact_features(test_data)

    predg = model.predict(test_features)
    name=classes[np.argmax(predg[0])]
    pre=round(np.max(predg[0])) * 100
    print(f"Predicted label: {classes[np.argmax(predg[0])]}")
    print(f"Probability of prediction): {round(np.max(predg[0])) * 100} %")
    
    # file_path = os.path.join('C:/Users/Administrator/Desktop/projet/static/uploads',file.filename)
    # rename_jpg_files(file_path,name)
    return redirect(url_for('home', file=file.filename,name=name,pre=pre))



if __name__ == '__main__':
    app.run(debug=False)
    





