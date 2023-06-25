import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import mlflow
import mlflow.keras
import keras.backend as K
from keras.optimizers import Adam
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

# Set the MLflow experiment name
mlflow.set_experiment("project")

# Set the root directory containing the data
root = "/app/data/"

# Read the images and store them in arrays
x = []
y = []
classnames = os.listdir(root)
for clas in classnames:
    img_path = os.path.join(root, clas)
    class_num = classnames.index(clas)
    for image in os.listdir(img_path):
        try:
            ip = os.path.join(img_path, image)
            img_array = cv.imread(ip, cv.IMREAD_COLOR)
            n_array = cv.resize(img_array, (256, 192))
            x.append(n_array)
            y.append(class_num)
        except:
            pass

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x, y, random_state=104, test_size=0.45, shuffle=True
)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

# Convert the labels to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Define the model architecture
pre_trained_model = DenseNet201(
    input_shape=(192, 256, 3), include_top=False, weights="imagenet"
)

for layer in pre_trained_model.layers:
    print(layer.name)
    if hasattr(layer, "moving_mean") and hasattr(layer, "moving_variance"):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False
last_layer = pre_trained_model.get_layer("relu")
print("last layer output shape:", last_layer.output_shape)
last_output = last_layer.output

# Compile the model
x = layers.GlobalMaxPooling2D()(last_output)
x = layers.Dense(512, activation="relu")(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(9, activation="softmax")(x)


model = Model(pre_trained_model.input, x)
optimizer = Adam(
    lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True
)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

# Start the MLflow run
with mlflow.start_run():
    # Enable MLflow autologging
    mlflow.keras.autolog()

    # Train the model
    batch_size = 24
    epochs = 1
    history = model.fit(
        x=np.array(x_train),
        y=np.array(y_train),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(np.array(x_val), np.array(y_val)),
        verbose=1,
        steps_per_epoch=(len(x_train) // batch_size),
        validation_steps=(len(x_val) // batch_size),
        callbacks=[ReduceLROnPlateau()],
    )

    # Save the model
    model.save("dense-net-finetuned")

    # Log the model as an artifact
    mlflow.log_artifact("dense-net-finetuned")

    # Evaluate the model
    # Evaluate the model
    loss_test, acc_test = model.evaluate(np.array(x_val), np.array(y_val), verbose=1)

    # Log metrics
    mlflow.log_metric("accuracy", acc_test)
    mlflow.log_metric("loss", loss_test)

    # Log the classification report
    y_pred = model.predict(np.array(x_val))
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(np.array(y_val), axis=1)

    # cm = confusion_matrix(y_true_labels, y_pred_labels)
    # cm_string = "\n".join([str(row) for row in cm])
    # mlflow.log_metric("confusion_matrix", cm_string)

    # # Calculate and log the classification report
    # clr = classification_report(y_true_labels, y_pred_labels, zero_division=1)
    # mlflow.log_metric("classification_report", clr)

# End the MLflow run
mlflow.end_run()
