BrainTumorClassification Project 
---------------------------------

This project classifies brain MRI images into two categories: yes (tumor present), and no (no tumor present).

It uses a Convolutional Neural Network built with TensorFlow and Keras and is trained on images organized into two folders: yes and no. The model learns to distinguish tumors from non-tumors.

How the project works: 
We have a folder called datasets which contains 2 subfolders: yes and no.
Each image in the yes folder shows an MRI image with a tumor and each image in the no folder shows an MRI image without a tumor. 
We read these images using Tensorflow's ImageDataGenerator and split them into training and validation sets (80% Training / 20% Validation). 
We use a CNN model with convolution and pooling layers and then we flatten the outputs and pass them through dense layers.
ReLU is the activation function we use which uses a sigmoid that outputs a probability between 0 and 1 representing a tumor or no tumor.
The model runs for 10 number of epochs and outputs the following metrics: 
- Train Accuracy and Train Loss 
- Validation Accuracy and Validation Loss 

After training the model we plot an accuracy and loss over the epochs so we can visually see how the model learns.
We also include single image prediction which allows loading a single image and predicting whether or not a tumor is present. 





