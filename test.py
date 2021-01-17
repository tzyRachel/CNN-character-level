import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint, Callback, ProgbarLogger)
from tensorflow.keras import backend as K
import data_handling_dna

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# test_file = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_spe_50to200_test_10k.tsv'

test_file = '/tmp/tzy/Metagenomic-Data/10_spe_50to200_test_10k.tsv'

seq_len = 200 # Fixed length of a sequence of chars, given
num_classes = 10 # Num of categories/concepts, given
alphabet = "ACGT" #alphabet set, given

model = load_model('model_200_big.hdf5', compile=False)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

#--------------- TESTING ------------------#

filename = test_file
data = data_handling_dna.read_data(filename,alphabet,seq_len,num_classes)
x_test = data[0]  # Testing input character sequences
y_test = data[1]	 # Testing input labels

# Testing
evl = model.evaluate(x_test, y_test, batch_size=1, verbose=2)
prediction = model.predict(x_test)
# over_half = 0
# for i in range(1000):
# 	if max(prediction[i]) > 0.5:
# 		over_half = over_half + 1
# print(over_half/1000)
	# print(prediction[i], np.sum(np.array(prediction[i])))
# Prints to console, the final loss and final accuracy averaged across all test samples
print(model.metrics_names[0],":",evl[0])
print(model.metrics_names[1],":",evl[1])
