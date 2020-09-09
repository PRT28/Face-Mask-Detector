import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

directory = r"D:\Pandas\FaceDetector\dataset"
categories = ["with_mask", "without_mask"]

data=[]
labels=[]

for category in categories:
    path= os.path.join(directory, category)
    for file in os.listdir(path):
        file_path= os.path.join(path,file)
        img= load_img(file_path, target_size=(224,224))
        img= img_to_array(img)
        img= preprocess_input(img)
        
        data.append(img)
        labels.append(category)

lb=LabelBinarizer()
labels= lb.fit_transform(labels)
labels= tf.keras.utils.to_categorical(labels)

data= np.array(data, dtype="float32")
labels=np.array(labels)

x_train, x_test, y_train, y_test= train_test_split(data, labels, test_size=0.2, stratify=labels)

augmentations= ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest"
                )

base= MobileNetV2(input_tensor=tf.keras.layers.Input(shape=(224,224,3)), include_top=False, weights="imagenet")

head= base.output
head= tf.keras.layers.AveragePooling2D(pool_size=(7,7))(head)
head= tf.keras.layers.Flatten()(head)
head= tf.keras.layers.Dense(128, activation='relu')(head)
head= tf.keras.layers.Dropout(0.5)(head)
head= tf.keras.layers.Dense(2, activation="softmax")(head)

model= tf.keras.Model(inputs=base.input, outputs=head)

lr=1e-4
epoch=20

optimize= tf.keras.optimizers.Adam(learning_rate= lr, decay=lr/epoch)
model.compile(loss="binary_crossentropy", optimizer=optimize, metrics=["acc"])

history= model.fit(
        augmentations.flow(x_train,y_train,32),
        steps_per_epoch= len(x_train)//32,
        validation_data=(x_test, y_test),
        validation_steps= len(x_test)//32,
        epochs=epoch)

pred= model.predict(x_test, batch_size=32)

pred= np.argmax(pred, axis=1)

print(classification_report(y_test.argmax(axis=1), pred, target_names= lb.classes_))

model.save("mask_detection.model", save_format="h5")

N = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")