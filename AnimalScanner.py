import keras
import os
import sys
import numpy as np
import cv2 as open
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message=".*iCCP: known incorrect sRGB profile.*")

CATEGORIES = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
EPOCHS = 25
TEST = 0.2

MODEL_SAVE = "animalScanner.keras"
def main():
    
    if len(sys.argv) > 2:
        sys.exit("BadInput:ERROR")
    
    images, labels = get_data(sys.argv[1]) 
    labels = keras.utils.to_categorical(labels)
    x_train,x_test,y_train,y_test = train_test_split(
        np.array(images),np.array(labels), test_size=TEST 
    )
    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,y_test,verbose=2)
    model.save(MODEL_SAVE)
   
def get_data(data_dir):
    images = []
    labels  = []
    for category in CATEGORIES:
        cater = os.path.join(data_dir,str(category))
        for img in os.listdir(cater):
            img_path = os.path.join(cater, img)
            image = open.imread(img_path)
            
            if image is not None:
                
                resize = open.resize(image, (64, 64))
                images.append(resize/255.0)
                labels.append(CATEGORIES.index(category)) 
    return images, labels
def get_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(CATEGORIES), activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()
    return model

main()

