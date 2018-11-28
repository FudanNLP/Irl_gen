import numpy as np


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        # self.batch_size = batch_size
        self.batch_size = batch_size // 2
        self.sentences = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.pos_sentences = np.array(positive_examples)
        self.neg_sentences = np.array(negative_examples)

        # Shuffle the data
        shuffle_pos_indices = np.random.permutation(np.arange(len(self.pos_sentences)))
        shuffle_neg_indices = np.random.permutation(np.arange(len(self.neg_sentences)))
        self.pos_sentences = self.pos_sentences[shuffle_pos_indices]
        self.neg_sentences = self.neg_sentences[shuffle_neg_indices]

        # Split batches
        self.num_batch = int(min(len(self.pos_sentences),len(self.neg_sentences)) / self.batch_size)
        self.pos_sentences = self.pos_sentences[:self.num_batch * self.batch_size]
        self.neg_sentences = self.neg_sentences[:self.num_batch * self.batch_size]
        self.pos_sentences_batches = np.split(self.pos_sentences, self.num_batch, 0)
        self.neg_sentences_batches = np.split(self.neg_sentences, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        texts = np.concatenate((self.pos_sentences_batches[self.pointer], self.neg_sentences_batches[self.pointer]), axis=0)
        self.pointer = (self.pointer + 1) % self.num_batch
        return texts

    def reset_pointer(self):
        self.pointer = 0

