import warnings
warnings.filterwarnings('ignore')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint, Callback, ProgbarLogger)
from tensorflow.keras import backend as K
import data_handling_dna

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# train_file = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_spe_50to200_train_10k.tsv'

train_file = '/tmp/tzy/Metagenomic-Data/10_spe_50to200_train_1m.tsv'

filename = train_file
seq_len = 200 # Fixed length of a sequence of chars
num_classes = 10 # Num of categories/concepts, given
init_step_size = 0.01 # Given
max_epochs = 30     # Num of epochs training
mini_batch_size = 256
alphabet = "ACGT" #alphabet set, given
step_size = init_step_size
data = data_handling_dna.read_data(filename,alphabet,seq_len,num_classes)
x = data[0] # Training input character sequences
y = data[1] # Training input labels

#Function to implement step size decay (halves every 3 epochs, 10 times)
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
#
# # Callbacks to save and retreive the best weight configurations found during training phase
all_callbacks = [step_size_callback, FlushCallback(),
						   ModelCheckpoint('model_200_big' + '.hdf5',
										   save_best_only=True,
										   verbose=1),
						   ProgbarLogger(count_mode='steps')]

#--------------- TRAINING ------------------#
# No vaidation split was given in the paper, so I've set it to 0.2 arbitarily
hist = model.fit(x, y, batch_size=mini_batch_size, epochs=max_epochs,
				 verbose=2, validation_split=0.2, callbacks=all_callbacks)
print(hist.history)
print("\nTraining completed. Beginning testing.\n")
