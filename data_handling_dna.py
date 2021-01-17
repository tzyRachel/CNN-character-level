import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import csv

LABEL_DICT = np.load('/tmp/tzy/Metagenomic-Data/label_dict.npy',
					 allow_pickle=True).item()

# LABEL_DICT = np.load('C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\label_dict.npy',
#   					 allow_pickle=True).item()

# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data(filename, alphabet, seq_len, num_classes):
	f = open(filename, 'r')
	# Create a list of the alphabet
	char_list = list(alphabet)
	all_seq_one_hot = []
	labels = []

	for line in f:
		l = line.strip().split('\t')
		seq = l[0]
		category = l[1]

		# Convert characters to character indices
		index_seq = []
		for i in seq:
			if i not in char_list:
				index = -1
			else:
				index = char_list.index(i)
			index_seq.append(index)

		# Padding with index '-1' if the sequence is not long enough
		# NOTE: '-1' is also used as the index of unknown characters (padding can be treated as unknown anyway)
		if len(index_seq) < seq_len:
			for _ in range(seq_len-len(index_seq)):
				index_seq.append(-1)

		# Convert indices to one-hot-vectors, '-1' becomes the all-zeros vector
		index_seq_one_hot = []
		for i in index_seq:
			all_zeroes = [0]*len(char_list)
			if i >= 0:
				all_zeroes[i] = 1
			index_seq_one_hot.append(all_zeroes)

		all_seq_one_hot.append(index_seq_one_hot)
		labels.append(category)

	x = np.array(all_seq_one_hot, dtype='float32')

	# Convert integer class labels to one-hot vectors
	all_labels_one_hot = []
	for i in labels:
		one_hot = [0]*num_classes
		one_hot[LABEL_DICT[i]] = 1
		all_labels_one_hot.append(one_hot)
	y = np.array(all_labels_one_hot, dtype='float32')

	return x, y

def generate_seq_tsv(min_len, max_len, num_seq, seq_dict):
	name_list = list(seq_dict.keys())
	i = 0
	file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_spe_'\
				+str(min_len)+'to'+str(max_len)+'_test_1m.tsv'

	with open(file_name, 'w', newline='') as f:
		tsv_w = csv.writer(f, delimiter='\t')
		while i < num_seq:
			name = random.choice(name_list)
			subname = random.choice(list(seq_dict[name].keys()))
			lenth = random.randint(min_len, max_len)
			start = random.randint(0, len(seq_dict[name][subname]) - min_len)
			seq = seq_dict[name][subname][start:start+lenth]
			name = name.replace('.fna', '')
			tsv_w.writerow([seq, name])
			i = i+1

if __name__ == '__main__':
	alphabet = "ACGT"
	seq_len = 200  # Fixed length of a sequence of chars, given
	num_classes = 10  # Num of categories/concepts, given
	file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_species.npy'
	seq_dictionary = np.load(file_name, allow_pickle=True).item()
	generate_seq_tsv(50, 200, 10**6, seq_dictionary)















