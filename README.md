# "Dog-Breed-Identification":
This project is a web application built with Flask for predicting the breed of a dog based on an uploaded image. The app utilizes four pre-trained deep learning models: InceptionV3, Xception, NASNetLarge, and InceptionResNetV2, to provide diverse and accurate predictions. Users can upload a dog image through the web interface, and the app will determine the most likely dog breed along with confidence scores using the ensemble of these models.
### Home Page
![image](https://github.com/oumaimamzoughi/Dog-Breed-Identification/assets/153776526/5f2b2ea4-8deb-41f1-bb05-f40ba5529481)
### Uploading an image of a dog
![image](https://github.com/oumaimamzoughi/Dog-Breed-Identification/assets/153776526/ccab8505-3773-44b2-b02f-80dc76a7d59d)
### Prediction using Deep Learning
![image](https://github.com/oumaimamzoughi/Dog-Breed-Identification/assets/153776526/2dd3907a-d9e5-44fb-8308-830b58c762f1)
## Steps Involved:
### file collab_dog_cognition.ipynb:
#### 1. Loading Libraries:
Importing necessary libraries and modules, such as TensorFlow, Keras, matplotlib, seaborn, etc.
#### 2. Reading Data:
Reading the labels from a CSV file (filtered_labels.csv) containing information about dog breeds.
#### 3.Data Exploration:
Exploring the data to understand the distribution of dog breeds and the number of images for each breed.

#### 4.Loading Images:
Loading and filtering image files from a specified directory (/content/drive/MyDrive/projetgod/train).

#### 5.Feature Extraction:
Extracting features from images using pre-trained deep learning models (InceptionV3, Xception, NASNetLarge, InceptionResNetV2).

#### 6.Data Preprocessing:
Preprocessing the data and combining the extracted features for model training.

#### 7.Model Building:
Building a neural network model using Keras with dense and dropout layers.

#### 8.Model Training:
Training the model using the extracted features and labels.

#### 9.Saving the Model:
Saving the trained model for future use.

#### 10. Loading Test Data:
Loading test data images for making predictions.

#### 11. Feature Extraction for Test Data:
Extracting features from test data using the same pre-trained models.

#### 12. Making Predictions:
Using the trained model to make predictions on the test data.

#### 13. Creating Submission File:
Creating a submission file in CSV format with predictions for each test image.

#### 14. Individual Prediction:
Demonstrating how to make predictions for a specific image .
