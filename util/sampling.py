import numpy as np
import copy
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MinibatchSampling:
    def __init__(self, array, batch_size, sim):
        if len(array) < batch_size:
            raise Exception('Length of array is smaller than batch size. len(array): ' + str(len(array))
                            + ', batch size: ' + str(batch_size))

        self.array = copy.deepcopy(array)   # So that the original array won't be changed
        self.batch_size = batch_size
        self.start_index = 0

        self.rnd_seed = (sim + 1) * 1000
        np.random.RandomState(seed=self.rnd_seed).shuffle(self.array)

    def get_next_batch(self):
        if self.start_index + self.batch_size >= len(self.array):
            self.rnd_seed += 1
            np.random.RandomState(seed=self.rnd_seed).shuffle(self.array)
            self.start_index = 0

        ret = [self.array[i] for i in range(self.start_index, self.start_index + self.batch_size)]
        self.start_index += self.batch_size

        return ret

def sampling(batch_size, train_image, train_label):

    train_image_batch = []
    train_label_batch = []
    batch_indices = np.random.choice(len(train_image), batch_size)

    for x in batch_indices: 
        train_image_batch.append(train_image[x])
        train_label_batch.append(train_label[x])

    train_image_batch = np.array(train_image_batch)
    train_label_batch = np.array(train_label_batch)
    train_image_batch = train_image_batch.reshape(100, 28, 28, 1)

    return train_image_batch, train_label_batch