#Authors: Tyvon Factor-Gaymon (6580310) and Peter Fung (6509830)
# The code should download the required file but if not, the CIFAR-10 dataset file can be downloaded at https://www.cs.toronto.edu/~kriz/cifar.html

import tensorflow as tf # Tensor flow utilizing python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
#from keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt
import math 

# Enable mixed precision training
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Check available devices
devices = tf.config.list_physical_devices()
print(devices)

# Set device to GPU if available, else use CPU
if len(devices) > 0 and 'GPU' in devices[0].device_type:
    device = '/GPU:0'
else:
    device = '/CPU:0'

# Begin Timer
start_time = time.time()

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
num_classes = 10 # CIFAR-10 = 10, CIFAR-100 = 100
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# Load pre-trained VGG16 model without top layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the weights of the pre-trained layers
for layer in vgg_model.layers:
    layer.trainable = False

# Add new top layers for your specific task
new_model = Sequential()
new_model.add(vgg_model)
new_model.add(Flatten())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dense(num_classes, activation='softmax'))

# Compile the model
opt = tf.keras.optimizers.Adam()
new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate schedule
def lr_schedule(epoch, total_epochs):
    fraction = epoch / total_epochs
    lr = 1e-3
    if fraction > 0.75:
        lr *= 0.1
    elif fraction > 0.5:
        lr *= 0.5
    elif fraction > 0.25:
        lr *= 0.75
    return lr

# Cyclic learning rate schedule
def cyclic_lr(epoch, base_lr, max_lr, step_size):
    cycle = math.floor(1 + epoch / (2 * step_size))
    x = abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
    return lr

# Set the initial learning rate and maximum learning rate
base_lr = 1e-4
max_lr = 1e-3

# Set the step size for the cyclic learning rate schedule
step_size = 5

# Create the LearningRateScheduler callback with the cyclic_lr function
#lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: cyclic_lr(epoch, base_lr, max_lr, step_size))

# Decay learning rate
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule) 

# Train the model
num_epochs = 120
batch_size = 128 # Use a larger batch size 128 256
steps_per_epoch = len(x_train) // batch_size
history = new_model.fit(datagen.flow(x_train, y_train_one_hot, batch_size=batch_size), # Augmented data
#history = new_model.fit(x_train, y_train_one_hot, batch_size=batch_size,
              steps_per_epoch=steps_per_epoch, 
              validation_data=(x_test, y_test_one_hot), 
              epochs=num_epochs, 
              callbacks=[lr_callback])

end_time = time.time()
elapsed_time_sec = end_time - start_time
elapsed_time_mins = (end_time - start_time)/60

print(f"Elapsed time: {elapsed_time_sec:.2f} seconds == {elapsed_time_mins:.2f} minutes")

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.text(num_epochs-1, history.history['accuracy'][-1], f"{history.history['accuracy'][-1]:.3f}", ha='center', va='bottom')
plt.text(num_epochs-1, history.history['val_accuracy'][-1], f"{history.history['val_accuracy'][-1]:.3f}", ha='center', va='top')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.text(num_epochs-1, history.history['loss'][-1], f"{history.history['loss'][-1]:.3f}", ha='center', va='bottom')
plt.text(num_epochs-1, history.history['val_loss'][-1], f"{history.history['val_loss'][-1]:.3f}", ha='center', va='top')
plt.show()

'''
Cyclic Augmented 60 epochs 32 batch
1562/1562 [==============================] - 11s 7ms/step - loss: 1.0940 - accuracy: 0.6117 - val_loss: 1.1232 - val_accuracy: 0.6083
Elapsed time: 638.87 seconds == 10.65 minutes
'''

