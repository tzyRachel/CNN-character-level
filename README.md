#####model.py:

Construct a character-level CNN model by tnesorflow. Currently, the model consists of 8 CNN layers with max-pooling and 4 fully connected layers
It saves a model named "model_200_big.hdf5"

#####train.py

Train the model_200_big.hdf5. 

Training set is '/tmp/tzy/Metagenomic-Data/10_spe_50to200_train_1m.tsv'. It consists of 1 million dna segments with length 50 to 200.

#####test.py
Test the model on the test set '/tmp/tzy/Metagenomic-Data/10_spe_50to200_test_10k.tsv'
Output average loss and accuracy

#####data_handling_dna.py contains 2 functions

read_data(filename, alphabet, seq_len, num_classes):
Reading data from training / testing files and returning one-hot-encodings to driver

generate_seq_tsv(min_len, max_len, num_seq, seq_dict): 
Generate a dataset in tsv form. It consists of "num_seq" dna segments with length 
from "min_len" to "max_len". 
