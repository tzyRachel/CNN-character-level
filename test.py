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

test_file = '/tmp/Metagenomic-Data/10_spe_50to200_test_10k.tsv'
print(test_file)

seq_len = 200 # Fixed length of a sequence of chars, given
num_classes = 10 # Num of categories/concepts, given
init_step_size = 0.01 # Given
max_epochs = 30     # Num of epochs training happens for - arbitarily set to 33 to observe step size decay
mini_batch_size = 256 # Given value is 128, but I've set to 1 to run quickly on toy data
momentum = 0.9 # Given
alphabet = "ACGT" #alphabet set, given
alph_size = len(alphabet)
step_size = init_step_size

def step_size_decay(epoch):
	if epoch > 1 and epoch <= 30 and epoch%3==1:
		global step_size
		step_size = step_size/2
	return step_size

#Function to print epoch count, loss and step size (to observe decay) after every epoch
class FlushCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		optimizer = self.model.optimizer
		print('Epoch %s: loss %s' % (epoch, logs.get('loss')))
		print("Step size:",K.eval(optimizer.lr))

model = load_model('model_200_big.hdf5', compile=False)
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

step_size_callback = LearningRateScheduler(step_size_decay)

# Callbacks to save and retreive the best weight configurations found during training phase
all_callbacks = [step_size_callback, FlushCallback(),
						   ModelCheckpoint('model_200_big' + '.hdf5',
										   save_best_only=True,
										   verbose=1),
						   ProgbarLogger(count_mode='steps')]

#--------------- TESTING ------------------#

filename = test_file
data = data_handling_dna.read_data(filename,alphabet,seq_len,num_classes)
x_test = data[0]  # Testing input character sequences
y_test = data[1]	 # Testing input labels

# Testing
evl = model.evaluate(x_test, y_test, batch_size=1, verbose=2)
prediction = model.predict(x_test)
over_half = 0
for i in range(1000):
	if max(prediction[i]) > 0.5:
		over_half = over_half + 1
print(over_half/1000)
	# print(prediction[i], np.sum(np.array(prediction[i])))
# Prints to console, the final loss and final accuracy averaged across all test samples
print(model.metrics_names[0],":",evl[0])
print(model.metrics_names[1],":",evl[1])
