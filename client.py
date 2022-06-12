import socket
import time
import random
import struct

from data_reader.data_reader import get_data, get_data_train_samples
from models.cnn_mnist2 import getCNNModel
from util.sampling import MinibatchSampling
from util.utils import send_msg, recv_msg
# Configurations are in a separate config.py file
from config import SERVER_ADDR, SERVER_PORT, dataset_file_path
import tensorflow as tf
import numpy as np


def initSetup(msg):

    batch_size_prev = None
    total_data_prev = None
    sim_prev = None

    dataset = msg[2]
    batch_size = msg[5]
    total_data = msg[6]
    indices_this_node = msg[8]
    read_all_data_for_stochastic = msg[9]
    sim = msg[11]

    # Assume the dataset does not change
    if read_all_data_for_stochastic or batch_size >= total_data:
        if batch_size_prev != batch_size or total_data_prev != total_data or (batch_size >= total_data and sim_prev != sim):
            print('Reading all data samples used in training...')
            train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

    batch_size_prev = batch_size
    total_data_prev = total_data
    sim_prev = sim

    if batch_size >= total_data:
        train_indices = indices_this_node
    else:
        train_indices = None  # To be defined later
    last_batch_read_count = None

    data_size_local = len(indices_this_node)
    w_prev_min_loss = None
    w_last_global = None
    total_iterations = 0

    send_msg(sock, ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER'])

    return (train_indices, last_batch_read_count, data_size_local, w_prev_min_loss, 
                w_last_global, total_iterations, train_image, train_label)

sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

try:
    while True:
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        # ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, num_iterations_with_same_minibatch_for_tau_equals_one, step_size, batch_size,
        # total_data, use_control_alg, indices_this_node, read_all_data_for_stochastic, use_min_loss, sim]

        dataset = msg[2]
        num_iterations_with_same_minibatch_for_tau_equals_one = msg[3]
        step_size = msg[4]
        batch_size = msg[5]
        total_data = msg[6]
        control_alg_server_instance = msg[7]
        read_all_data_for_stochastic = msg[9]
        use_min_loss = msg[10]

        (train_indices, last_batch_read_count, data_size_local, w_prev_min_loss, 
            w_last_global, total_iterations, train_image, train_label) = initSetup(msg)

        train_image = np.array(train_image)
        train_label = np.array(train_label)
        print("Train Image Shape: ", train_image.shape)
        print("Train Label Shape: ", train_label.shape)
        # for image_Index in range(0, len(train_image)):
        #     image = train_image[image_Index]
        #     sublists = [image[x:x+28] for x in range(0, len(image), 28)]
        #     train_image[image_Index] = sublists
        # print(train_image.shape)

        data = []
        for i in range(0, len(train_image)):
            data.append([train_image[i], train_label[i]])
        model = getCNNModel(step_size)

        while True:
            print('---------------------------------------------------------------------------')

            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
            w = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]
            prev_loss_is_min = msg[4]

            model = getCNNModel(step_size)
            model.summary()
            print("Setting Weights", end="\r")
            model.set_weights(w)
            print("----Weights set----")
            if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
                w_prev_min_loss = w_last_global

            time_local_start = time.time()  #Only count this part as time for local iteration because the remaining part does not increase with tau

            # Perform local iteration
            loss_last_global = None   # Only the loss at starting time is from global model parameter
            loss_w_prev_min_loss = None

            tau_actual = 0

            for i in range(0, tau_config):

                # When batch size is smaller than total data, read the data here; else read data during client init above
                # if batch_size < total_data:
                    # When using the control algorithm, we want to make sure that the batch in the last local iteration
                    # in the previous round and the first iteration in the current round is the same,
                    # because the local and global parameters are used to
                    # estimate parameters used for the adaptive tau control algorithm.
                    # Therefore, we only change the data in minibatch when (i != 0) or (sample_indices is None).
                    # The last condition with tau <= 1 is to make sure that the batch will change when tau = 1,
                    # this may add noise in the parameter estimation for the control algorithm,
                    # and the amount of noise would be related to NUM_ITERATIONS_WITH_SAME_MINIBATCH.

                    # if (i != 0) or (train_indices is None) or (tau_config <= 1 and
                    #             (last_batch_read_count is None or
                    #              last_batch_read_count >= num_iterations_with_same_minibatch_for_tau_equals_one)):

                # sample_indices = sampler.get_next_batch()

                # if read_all_data_for_stochastic:
                #     train_indices = sample_indices
                # else:
                #     train_image, train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
                #     train_indices = range(0, min(batch_size, len(train_label)))

                train_image_batch = []
                train_label_batch = []
                batch_indices = np.random.choice(len(data), batch_size)
                for x in batch_indices: 
                    train_image_batch.append(train_image[x])
                    train_label_batch.append(train_label[x])
                train_image_batch = np.array(train_image_batch)
                train_label_batch = np.array(train_label_batch)
                train_image_batch = train_image_batch.reshape(100, 28, 28, 1)

                if i == 0:
                    train_pred_batch = model.predict(train_image_batch)
                    score = model.evaluate(train_image_batch, train_label_batch, verbose=0)
                    loss_last_global = score[0]

                    w_last_global = w

                    if use_min_loss:
                        if (batch_size < total_data) and (w_prev_min_loss is not None):
                            # Compute loss on w_prev_min_loss so that the batch remains the same
                            model2 = getCNNModel(step_size)
                            model2.set_weights(w_prev_min_loss)
                            train_pred2 = model2.predict(train_image_batch, train_label_batch)
                            loss_w_prev_min_loss = model2.compute_loss(train_image_batch, train_label_batch, train_pred2)

                print("Training model", end="\r")
                model.set_weights(w)
                model.fit(train_image_batch, train_label_batch)
                w = model.get_weights()
                print("Model Trained (",i,")")

                tau_actual += 1
                total_iterations += 1

            # Local operation finished, global aggregation starts
            time_local_end = time.time()
            time_all_local = time_local_end - time_local_start
                
            msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                   loss_last_global, loss_w_prev_min_loss]
            send_msg(sock, msg)

            if is_last_round:
                break

except (struct.error, socket.error):
    print('Server has stopped')
    pass
