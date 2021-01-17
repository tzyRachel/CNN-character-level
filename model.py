import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Convolution1D, Input, MaxPooling1D, Flatten, Dense, Dropout)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

seq_len = 200 # Fixed length of a sequence of chars
num_classes = 10 # Num of categories
init_step_size = 0.01 # Given
max_epochs = 30     # Num of epochs training happens
mini_batch_size = 256
momentum = 0.9
alphabet = "ACGT" #alphabet set
alph_size = len(alphabet)
step_size = init_step_size

#--------------- ConvNet STRUCTURE ------------------#

#Input one-hot encoded sequence of chars
sequence_one_hot = Input(shape=(seq_len,alph_size),dtype='float32')

# 1st CNN layer with max-pooling
conv1 = Convolution1D(256,5,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(sequence_one_hot)
pool1 = MaxPooling1D(pool_size=3)(conv1)

# 2nd CNN layer with max-pooling
conv2 = Convolution1D(256,5,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=3)(conv2)

# 3rd CNN layer without max-pooling
conv3 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(pool2)

# 4th CNN layer without max-pooling
conv4 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv3)

# 5th CNN layer without max-pooling
conv5 = Convolution1D(256,3,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv4)

# 6th CNN layer with max-pooling
conv6 = Convolution1D(256,1,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv5)
pool6 = MaxPooling1D(pool_size=3)(conv6)

# 7th CNN layer with max-pooling
conv7 = Convolution1D(256,1,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv6)
pool7 = MaxPooling1D(pool_size=3)(conv7)

# 7th CNN layer with max-pooling
conv8 = Convolution1D(256,1,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(conv7)
pool8 = MaxPooling1D(pool_size=3)(conv8)

# Reshaping to 1D array for further layers
flat = Flatten()(pool8)

# 1st fully connected layer with dropout
dense1 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(flat)
dropout1 = Dropout(0.5)(dense1)

# 2nd fully connected layer with dropout
dense2 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)

# 3rd fully connected layer with dropout
dense3 = Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='relu')(dropout2)
dropout3 = Dropout(0.5)(dense3)

# 4th fully connected layer with softmax outputs
dense4 = Dense(10, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05), bias_initializer=RandomNormal(mean=0.0, stddev=0.05),activation='softmax')(dropout3)

# SGD: the learning rate set here doesn't matter because it gets overridden by step_size_callback
sgd = SGD(lr=init_step_size, momentum=momentum)

model = Model(sequence_one_hot, dense4)

model.save('model_200_big.hdf5')

