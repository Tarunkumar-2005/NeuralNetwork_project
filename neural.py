import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten
#Loading the data Fashion MNIST dataset
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
#displaying ths sample images
def display_samples(images,labels,num_samples=5):
    plt.figure(figsize=(10,2))
    for i in range(num_samples):
        plt.subplot(1,num_samples,i+1)
        plt.imshow(images[i],cmap='gray')
        plt.title(f'Label:{labels[i]}')
        plt.axis('off')
    plt.show()
display_samples(train_images,train_labels)
#pre processing the data
train_images=train_images.reshape((60000,28,28,1)).astype('float32') /255
test_images=test_images.reshape((10000,28,28,1)).astype('float32') /255
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

#Creating a neural network with 2 hidden layers
model=Sequential()
model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compiling model with optimizer,loss,metrics
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Training the model
model.fit(train_images,train_labels,epochs=5,batch_size=64,validation_split=0.2)

#evaluating the model on the test data
test_loss,test_acc=model.evaluate(test_images,test_labels)
print(f'the test accuracy will b:{test_acc}')

#predictions on a few test images
predictions=model.predict(test_images[:5])
predicted_labls=np.argmax(predictions,axis=1)
actual_labels=np.argmax(test_labels[:5],axis=1)

#display the test images and their Predictions
def display_predictions(images,actual,predicted):
    plt.figure(figsize=(10,2))
    for i in range(len(images)):
        plt.subplot(1,len(images),i+1)
        plt.imshow(images[i].reshape(28,28),cmap='gray')
        title=f'Actual:{actual[i]}\nPredicted:{predicted[i]}'
        plt.title(title)
        plt.axis('off')
    plt.show()

display_predictions(test_images[:5],actual_labels,predicted_labls)