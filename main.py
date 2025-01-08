#Amitoj's BrainTumorClassification Project

#imports 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 



def main():

    #PREPARING THE DATA 
    data_dir = "datasets" #path to folder containing subfolders "yes" and "no". "yes" folder containing images of mri scans with brain tumors and "no" containing images of mri scans with no brain tumors 

    #ImageDataGenerator will help load images 
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, #Normalizes pixel values
        validation_split=0.2  #We use 80% of images for training and 20% for validation 
    )

    #load images for training
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir, 
        target_size=(150, 150), #resize all images to 150x150
        batch_size=32,
        class_mode='binary', #using binary (1=yes, 0 = no)
        subset='training',
        shuffle=True #randomizes input 
    )

    #loading images for validation
    val_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation', #using 20% of the images for validation 
        shuffle=False #choosing to not randomize 
    )

    #printing indices to indicate how yes and no are mapped to labels.
    print("Class indices:", val_generator.class_indices)  
    

    #DEFINING THE MODEL 
    #defining sequential CNN model 
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), #32 filters each being 3x3. ReLu activation. Make each input image 150x150 with 3 color channels
        layers.MaxPooling2D((2, 2)), #reducing spacial dimensions by taking max value 

        layers.Conv2D(64, (3, 3), activation='relu'), #64 filters and 3x3. ReLu activation 
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'), #128 filters and 3x3. ReLu activation
        layers.MaxPooling2D((2, 2)), 

        layers.Flatten(), #.Flatten turns 3D into 1D feature vector 
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), #This randomly drops %50 of the connections to prevent overfitting 
        layers.Dense(1, activation='sigmoid') 
    ])

    #Compile the model 
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'] #tracking accuracy as the metric 
    )

    model.summary() #prints summary of the model layers, shapes, and parameters 

    #TRAINING THE MODEL 
    history = model.fit(
        train_generator,
        epochs=10, #train for 10 epochs
        validation_data=val_generator
    )

    #PLOTTING 
    #plotting the training and validation accuracy and loss to see how model learns
    plt.figure(figsize=(12, 4))


    #accuracy plot
    plt.subplot(1, 2, 1) #the first subplot and it is 1 row 2 columns.
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')  #graph title 
    plt.xlabel('Epoch') #labelling x axis
    plt.ylabel('Accuracy') #labelling y axis 
    plt.legend() #shows training accuracy and validation accuracy 

    #loss plot 
    plt.subplot(1, 2, 2) #the second subplot and it is 1 row 2 columns
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.tight_layout()
    plt.show() #adjusting layout and displaying plot

    #EVALUATION
    val_generator.reset() #reset the validation generator 
    preds = model.predict(val_generator) #get probabilities for each validation image  
    predicted_classes = (preds > 0.5).astype(int).ravel()

    
    true_classes = val_generator.classes
    #shows class labels such as yes or no
    class_labels = list(val_generator.class_indices.keys())

    #shows metrics 
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    #shows correct and incorrect predictions 
    print("Confusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))


    #SINGLE IMAGE PREDICTING
    #putting path to a single image for testing 
    single_img_path = "/Users/amitojrai/Documents/BrainTumorClassification/datasets/no/no9.jpg"  

    #if the image exists we pass it on to the function 
    if os.path.exists(single_img_path):
        predict_single_image(model, single_img_path)
    else: #if the image does not exist we print error message 
        print(f"Single image path not found: {single_img_path}") 


def predict_single_image(model, image_path):
    #Looks at a  single image and prints yes/no brain tumor prediction.
    
    #using OpenCV to read image 
    img = cv2.imread(image_path)

    #if the image fails to load print error message
    if img is None:
        print(f"Failed to load image from {image_path}")
        return

    #resize image to match the size we used when training
    img = cv2.resize(img, (150, 150))
    #convert and normalize
    img = img.astype('float32') / 255.0
    #adding batch dimension
    img = np.expand_dims(img, axis=0)

    #passing the image through the model 
    prediction = model.predict(img)[0][0]
    #compare prediction  probability to the threshold to se if yes or no 
    if prediction > 0.5:
        print(f"Prediction for '{image_path}': YES (tumor)  [Score: {prediction:.4f}]")
    else:
        print(f"Prediction for '{image_path}': NO  (no tumor) [Score: {prediction:.4f}]")

if __name__ == "__main__":
    main()