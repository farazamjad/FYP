#!/usr/bin/env python
# coding: utf-8
# pylint: disable=line-too-long
# pylint: disable=C0303
# pylint: disable=C0325
# pylint: disable=W0611
# pylint: disable=W0621
# pylint: disable=C0411
# pylint: disable=W0404
# pylint: disable=E0401
# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=R1732
# pylint: disable=R1705
# pylint: disable=E1101
# pylint: disable=C0305
# pylint: disable=W0104
# pylint: disable=W0702
# pylint: disable=C0114
# pylint: disable=R0914
# pylint: disable=C0412
# pylint: disable=C0413
# pylint: disable=C0209

import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.densenet import DenseNet201
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

import matplotlib.pyplot as plt



# In[11]:




# In[3]:


from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow
# pylint: disable=line-too-long


# In[13]:


import cv2 as cv
from sklearn.model_selection import train_test_split



# In[15]:


root="C:/Users/ahmad/Music/data/"


# In[142]:


#reading the images and storing in array to split into test and val
x=[]
y=[]

classnames=os.listdir(root)
for clas in classnames :
    img_path=os.path.join(root,clas)
    class_num=classnames.index(clas)
    for image in os.listdir(img_path):
        #print(os.path.join(img_path,image))
        try:
            ip=img_path+'/'+image    
            print(ip)
            img_array=cv.imread(ip,cv.IMREAD_COLOR)
            print(img_array)
            n_array=cv.resize(img_array,(256,192))
            x.append(n_array)
            y.append(class_num)
        except:
            pass
        
        


# In[ ]:





# In[143]:


x_train, x_val,y_train, y_val = train_test_split(x,y ,
                                   random_state=104, 
                                   test_size=0.45, 
                                   shuffle=True)


# In[144]:


x_train=np.array(x_train)
y_train=np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)


# In[145]:


y_train=to_categorical(y_train)


# In[146]:


y_val=to_categorical(y_val)


# In[21]:


print(x_train.shape)


# In[22]:


pre_trained_model = DenseNet201(input_shape=(192, 256, 3), include_top=False, weights="imagenet")


# In[23]:


for layer in pre_trained_model.layers:
    print(layer.name)
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False

print(len(pre_trained_model.layers))


# In[24]:


last_layer = pre_trained_model.get_layer('relu')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[28]:


x = layers.GlobalMaxPooling2D()(last_output)
x = layers.Dense(512, activation='relu')(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(9, activation='softmax')(x)


model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[29]:


model.summary()


# In[30]:


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(x_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_val)


# In[32]:


batch_size = 32
epochs = 6
history = model.fit(train_datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(x_val, y_val),
                              verbose = 1, steps_per_epoch=(x_train.shape[0] // batch_size), 
                              validation_steps=(x_val.shape[0] // batch_size))


# In[33]:


pre_trained_model.layers[481].name


# In[34]:


for layer in pre_trained_model.layers[481:]:
    layer.trainable = True


# In[35]:


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[36]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)


# In[37]:


model.summary()


# In[39]:


batch_size = 32
epochs = 30
history = model.fit(train_datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(x_val, y_val),
                              verbose = 1, steps_per_epoch=(x_train.shape[0] // batch_size),
                              validation_steps=(x_val.shape[0] // batch_size),
                              callbacks=[learning_rate_reduction])


# In[40]:


model.save('dense-net-finetuned')


# In[120]:


from tensorflow.keras.preprocessing.image import img_to_array,load_img


# In[164]:


img=load_img("C:/Users/ahmad/Music/data/Acne cyst/2 (2).jpg",target_size=(192,256))


# In[165]:


x=img_to_array(img)


# In[166]:


x=x.reshape((1,192,256,3))


# In[167]:


b=model.predict(x)


# In[168]:


print(classnames[np.argmax(b)])


# In[41]:


layer_outputs=[layer.output for layer in model.layers[1:]]


# In[93]:


visualize_model=tensorflow.keras.models.Model(inputs=model.input,outputs=layer_outputs)


# In[161]:


img=load_img("C:/Users/ahmad/Music/data/Acne cyst/2 (2).jpg",target_size=(192,256))


# In[162]:


x=img_to_array(img)


# In[163]:


x.shape


# In[151]:


loss_test, acc_test = model.evaluate(x_val, y_val, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (acc_test, loss_test))


# In[152]:


print(y_val.shape)


# In[158]:


import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import f1_score,confusion_matrix,classification_report


# In[ ]:





# In[159]:


def predictor(Cn,y_val,x_val):    
    y_pred= []
    error_list=[]
    error_pred_list = []
    y_true=[]
    classes=Cn
    class_count=len(classes)
    errors=0
    preds=model.predict(x_val, verbose=1)
    tests=len(preds)    
    for i, p in enumerate(preds):        
        pred_index=np.argmax(p)        
        true_index=np.argmax(y_val[i])  # labels are integer values 
        
        if pred_index != true_index: # a misclassification has occurred                                          
            errors=errors + 1
            file='oo'
            error_list.append(file)
            error_class=classes[pred_index]
            error_pred_list.append(error_class)
        y_true.append(true_index)
        y_pred.append(pred_index)
           
    acc=( 1-errors/tests) * 100
    msg=f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}'
    print(msg) # cyan foreground
    ypred=np.array(y_pred)
    ytrue=np.array(y_true)
    f1score=f1_score(ytrue, ypred, average='weighted')* 100
    if class_count <=30:
        cm = confusion_matrix(ytrue, ypred )
        # plot the confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)      
        plt.xticks(np.arange(class_count)+.5, classes, rotation=90)
        plt.yticks(np.arange(class_count)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes, digits= 4) # create classification report
    print("Classification Report:\n----------------------\n", clr)
    return errors, tests, error_list, error_pred_list, f1score

errors, tests, error_list, error_pred_list, f1score =predictor(classnames,y_val,x_val)


# In[6]:


classnames=['Acne cyst','Acne other','Actinic Keratosis and other Malignant Lesions',' basal cell','Eczema Photos','Melanoma Skin Cancer Nevi and Moles','Nail Fungus and other Nail Disease','Psoriasis','ROSACEA',]


# In[110]:


x=x.reshape((1,192,256,3))


# In[111]:


x=x/255


# In[112]:


featuremaps=visualize_model.predict(x)


# In[2]:


import tensorflow as tf


# In[3]:


model_path = "C:/Users/icl/Desktop/New folder/dense-net-finetuned"
model = tf.saved_model.load(model_path)


# In[20]:


import tensorflow as tf
 
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users/icl/Desktop/New folder/dense-net-finetuned") 
tflite_model = converter.convert()
 
# Save the model.
with open('fruits_model.tflite', 'wb') as f:
    f.write(tflite_model)


# In[4]:


from tensorflow.keras.preprocessing.image import img_to_array,load_img


# In[19]:


img=load_img('C:/Users/icl/Desktop/t3.jpeg',target_size=(192,256))
x=img_to_array(img)
x=x.reshape((1,192,256,3))
b=model(x)
print(classnames[np.argmax(b)])


# In[61]:


tf.__version__


# In[85]:


import matplotlib.pyplot as plt
import os
import cv2
# Set the path to the folder containing images
folder_path = 'C:/Users/icl/Desktop/imagesforfyp/'
fig = plt.figure(figsize=(100, 8))
# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
i=0
# Loop through the image files and plot them using matplotlib
for image_file in image_files:
    # Load the image using matplotlib
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    image = plt.imread(os.path.join(folder_path, image_file))
    label="test_image"
    ax.set_title(label)
     
    plt.imshow(image)
    i+=1

    # Plot the image
    
plt.show()
#This code uses the os module to get a list of all the image files in a folder, and then uses matplotlib to load and display each image. Note that this code assumes that the images in the folder are in either JPG or PNG format. You can modify the code to handle other image formats as well.







# In[57]:





# In[58]:


import numpy as np


# In[62]:


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score


# In[63]:


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size
     


# In[64]:


def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')


# In[65]:


TF_LITE_MODEL_FILE_NAME = "tf_lite_model.tflite"
     

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tf_lite_converter.convert()
     

tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)
     

convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), "KB")
     


# In[70]:


interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])


# In[71]:


test_imgs_numpy = np.array(x, dtype=np.float32)
     


# In[72]:


test_imgs_numpy.shape


# In[73]:


interpreter.set_tensor(input_details[0]['index'], test_imgs_numpy)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)
prediction_classes = np.argmax(tflite_model_predictions, axis=1)
     


# In[74]:


print(prediction_classes)


# In[ ]:


classnames=['Acne cyst','Acne other','Actinic Keratosis and other Malignant Lesions',' basal cell','Eczema Photos','Melanoma Skin Cancer Nevi and Moles','Nail Fungus and other Nail Disease','Psoriasis','ROSACEA',]
img=load_img('C:/Users/icl/Desktop/download.jpg',target_size=(192,256))
x=img_to_array(img)
x=x.reshape((1,192,256,3))
x=np.array(x, dtype=np.float32)
b=model(x)
print(classnames[np.argmax(b)])

