import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from random import shuffle

DIR = os.listdir('C:/Users/DELL/Documents/TestCv/COVID-19_Radiography_Dataset_1')
print(DIR)
# Define directories
train_folder = 'C:/Users/DELL/Documents/TestCv/COVID-19_Radiography_Dataset_1/train'
test_folder = 'C:/Users/DELL/Documents/TestCv/COVID-19_Radiography_Dataset_1/test'
val_folder = 'C:/Users/DELL/Documents/TestCv/COVID-19_Radiography_Dataset_1/val'

labels = ["normal", "viral","covid"]
IMG_SIZE = 224
NUM_SAMPLES = 6562
BATCH_SIZE = 6
def get_data(data_dir, num_samples=None):
    data = []
    num_samples_per_class = num_samples // 2 if num_samples else None
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        images = os.listdir(path)
        if num_samples_per_class:
            images = images[:num_samples_per_class]  # Balance the dataset
        for img in images:
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                print(e)
    shuffle(data)
    return np.array(data, dtype=object)

# Load the data
train = get_data(train_folder, num_samples=NUM_SAMPLES)
test = get_data(test_folder)
val = get_data(val_folder)

# Prepare the datasets
X_train = np.array([i[0] for i in train]) / 255.0
y_train = np.array([i[1] for i in train])
#z_train = np.array([i[1] for i in train])
X_val = np.array([i[0] for i in val]) / 255.0
y_val = np.array([i[1] for i in val])
#z_val = np.array([i[1] for i in val])
X_test = np.array([i[0] for i in test]) / 255.0
y_test = np.array([i[1] for i in test])
#z_test = np.array([i[1] for i in test])
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
#print(f"z_train shape: {z_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
#print(f"z_train shape: {z_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
#print(f"z_train shape: {z_test.shape}")

# Plot the distribution of training labels
sns.countplot(x=[labels[int(i)] for i in y_train])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Training Labels')
plt.show()

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val)

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D
print(f"X_train shape: {X_train.shape}")

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), shuffle=True, batch_size= BATCH_SIZE, callbacks=[callback])
scores = model.evaluate(X_test, y_test)

model.save("cnn_model.bin")