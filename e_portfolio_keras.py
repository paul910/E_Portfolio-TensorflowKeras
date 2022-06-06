# --------------------------------------------
# Handgeschriebende Ziffern identifizieren
# --------------------------------------------

# import von TF
import tensorflow as tf
print(tf.__version__)

# Handgeschriebene Ziffern laden
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(6,8, subplot_kw=dict(xticks=[], yticks=[]), figsize=(8,6))
for i, axi in enumerate(ax.flat):
  axi.imshow(x_train[1250*i], cmap='gray_r')

#Normalisierung (Werte zwischen 0 und 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_train[0])

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# TF Zahlenerkennungsmodell
model1 = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(512, activation='sigmoid'),
  Dropout(0.2),
  Dense(10, activation='softmax')
])

model1.summary()

# Crossentropy für die 10 Zahlen Klassen
model1.compile(optimizer='SGD', # Gradient Decent -> besser Adam Algorithm
              loss='sparse_categorical_crossentropy', # Berechnet den Kreuzentropieverlust zwischen den Labels und den Vorhersagen.
              metrics=['accuracy'])

# Modellfitting und Evaluation
fit_history = model1.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

print('Performance on test data: ', model1.evaluate(x_test, y_test))

plt.figure(1, figsize = (15,8))
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']) ;
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']);

predictions = model1.predict([x_test])
print(predictions[0])

import numpy as np
print(np.argmax(predictions[0]))

print(y_test[0])
plt.imshow(x_test[0], cmap="gray_r")

model1.save("model1")

model2 = keras.models.load_model("model1")

predictions = model2.predict([x_test])
print(np.argmax(predictions[1]))
plt.imshow(x_test[1], cmap='gray_r')

# --------------------------------------------
# Klassifikation von Tieren und Fahrzeugen
# --------------------------------------------

cifar10 = tf.keras.datasets.cifar10

# Aufteilung in Training- und Testset
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

label_names = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']

fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]), figsize=(10,8))
fig.tight_layout(pad=2)
for i, axi in enumerate(ax.flat):
  axi.imshow(x_train[1000*i], cmap='gray_r')
  axi.set_title(label_names[y_train[1000*i][0]])
print(x_train.shape)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model2 = Sequential([
  Flatten(input_shape=(32, 32, 3)),
  Dense(512, activation=tf.nn.sigmoid),
  Dropout(0.2),
  Dense(10, activation=tf.nn.softmax)
])
 
# Crossentropy für die 10 Zahlen Klassen
model2.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
# Modellfitting und Evaluation
fit_history = model2.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print('')
print('Performance on test data: ', model2.evaluate(x_test, y_test))

plt.figure(1, figsize = (15,8))
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']) ;
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']);

# --------------------------------------------
# Convolutional Neural Network
# --------------------------------------------

from tensorflow.keras.layers import Conv2D, MaxPooling2D

num_filters = 8
filter_size = 3
pool_size = 2

# TF Bilderkennungsmodell
model3 = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(32, 32, 3), activation='relu'),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),  
])
 
# Crossentropy für die 10 Zahlen Klassen
model3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
# Modellfitting und Evaluation
fit_history_3 = model3.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model3.evaluate(x_test, y_test)

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history_3.history['accuracy'])  
plt.plot(fit_history_3.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']) 
    
plt.subplot(222)  
plt.plot(fit_history_3.history['loss'])  
plt.plot(fit_history_3.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'validation']);