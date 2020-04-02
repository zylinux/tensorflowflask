#!/usr/bin/python3
#please run this with command (python3 testcla.py)
#if you do not have python3 please do sudo apt-get upgrade python3

# TensorFlow and tf.keras, make sure you did 
#sudo apt-get install python3-dev python3-pip
#sudo pip3 install --upgrade pip3
#sudo pip3 install --upgrade tensorflow
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def functionbbb():
  	# Something
	print(tf.__version__)

	#download data
	fashion_mnist = keras.datasets.fashion_mnist

	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	#how many classes
	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	#train data images shape and length
	train_images.shape
	len(train_labels)
	train_labels

	#train data labels,given to us by default
	test_images.shape
	len(test_labels)

	#need float number
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	#Build the model
	#setup layers 
	model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),keras.layers.Dense(128,activation='relu'),keras.layers.Dense(10)])

	#compile model
	model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

	#train model
	model.fit(train_images, train_labels, epochs=10)

	#evaluate accuracy
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print('\nTest accuracy:', test_acc)

	#With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(test_images)
	predictions[0]
	#predications[0] is the first test data we using model did the prediction, lets see  label is x
	np.argmax(predictions[0])
	
	#lets confirm it real label x , if np.argmax(predictions[0]) ==  test_labels[0] means we are right.
	test_labels[0]

	#######################################################################################################
	# Grab an image from the test dataset.
	img = test_images[1]
	print(img.shape)

	# Add the image to a batch where it's the only member.
	img = (np.expand_dims(img,0))
	print(img.shape)	

	#Now predict the correct label for this image:
	predictions_single = probability_model.predict(img)
	print(predictions_single)

	#tell us which label it will be 
	ret = np.argmax(predictions_single[0])
	print(ret)
	print(class_names)
	return ret

