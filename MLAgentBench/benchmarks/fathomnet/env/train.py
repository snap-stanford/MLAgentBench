import ast

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization
import json

def label_map(category, n_classes=290):
    category = ast.literal_eval(category)
    labels = [0]*n_classes
    for category_id in category:
        labels[int(category_id)-1] = 1
    return labels

if __name__ == "__main__":

    ## load data

    image_dir = "images/" 

    train_df = pd.read_csv("multilabel_classification/train.csv")
    train_df['categories'] = train_df['categories'].apply(label_map)
    file_name = []

    for idx in range(len(train_df)):
        file_name.append(image_dir + train_df["id"][idx]+".png")
            
    train_df["file_name"] = file_name
        
    X_dataset = []  

    SIZE = 256

    for i in range(len(train_df)):
        img = keras.utils.load_img(train_df["file_name"][i], target_size=(SIZE,SIZE,3))
        img = keras.utils.img_to_array(img)
        img = img/255.
        X_dataset.append(img)
        
    X = np.array(X_dataset)
    y = np.array(train_df["categories"].to_list())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)

    # define model

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(SIZE,SIZE,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(290, activation='sigmoid'))

    # Train model

    EPOCH = 1
    BATCH_SIZE = 64

    #Binary cross entropy of each label. So no really a binary classification problem but
    #Calculating binary cross entropy for each label. 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_test, y_test), batch_size=BATCH_SIZE)
 
    ## generate predictions on test set and save to submission.csv
    valid_json = json.load(open("object_detection/eval.json"))["images"]
    valid_df = pd.DataFrame(valid_json)

    predict_list = []

    for i in range(len(valid_df)):

        img = keras.utils.load_img(image_dir + valid_df['file_name'][0], target_size=(SIZE,SIZE,3))
        img = keras.utils.img_to_array(img)
        img = img/255.
        img = np.expand_dims(img, axis=0)

        classes = np.array(pd.read_csv("category_key.csv")["name"].to_list()) #Get array of all classes
        proba = model.predict(img)  #Get probabilities for each class
        sorted_categories = np.argsort(proba[0])[:-11:-1]  #Get class names for top 10 categories
        
        threshold = 0.5
        predict = []
        proba = proba[0]
        for i in range(len(proba)):
            if proba[i]>=threshold:
                predict.append(i+1) #Get true id of the class
        predict.sort()
        predict_list.append(predict)

    valid_id = [x[:-4]  for x in valid_df["file_name"].to_list()]
    valid_osd = [1]*len(valid_id)

    submit_data = [[valid_id[i], predict_list[i], valid_osd[i]] for i in range(len(valid_id))]
    pd.DataFrame(data=submit_data, columns=["id", "categories", "osd"]).to_csv("submission.csv", index=False)

