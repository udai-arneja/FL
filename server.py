import socket
import time
import numpy as np
import argparse

from data_reader.data_reader import get_data
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case
import matplotlib.pyplot as plt
from models.cnn_mnist2 import getCNNModel

# Configurations are in a separate config.py file
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--one',action='store_true')
args=parser.parse_args()

if args.one:
    tau_setup_all = [1]

# start up sockets to interact with clients - returns list of socket
def socketsInFLSystem(): 
    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all=[]

    # Establish connections to each client, up to n_nodes clients
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(5)
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        print(client_sock)  # compilation of info about the socket

        client_sock_all.append(client_sock)
    
    return client_sock_all

# model import, time generation, data organisation
def setup():

    if time_gen is not None:
        use_fixed_averaging_slots = True
    else:
        use_fixed_averaging_slots = False

    if batch_size < total_data:   # Read all data once when using stochastic gradient descent
        train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)

        # This function takes a long time to complete,
        # putting it outside of the sim loop because there is no randomness in the current way of computing the indices
        indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)
    
    return [use_fixed_averaging_slots, train_image, train_label, test_image, test_label, train_label_orig, indices_each_node_case]

# single or multi sim paths setup
def simulationBegin(single_run):
    if single_run:
        stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
    else:
        stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)
    return stat

def variablesSetup():
    time_global_aggregation_all = None

    total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
    total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
                                # using predefined values when use_fixed_averaging_slots = true
    it_each_local = None
    it_each_global = None

    is_last_round = False
    is_eval_only = False

    tau_new_resume = None

    return  (time_global_aggregation_all, total_time, total_time_recomputed, it_each_local, 
            it_each_global, is_last_round, is_eval_only, tau_new_resume)

def noiseGeneration(w, noiseDist):
    npNoiseArray = []
    for rowIndex in range(len(w)):
        row = w[rowIndex]
        mean = np.mean(row)
        var = np.var(row)
        npNoise = np.random.normal(mean, var, row.shape) * noiseDist
        # print(npNoise)
        npNoiseArray.append(npNoise)
    return npNoiseArray

def evalModel(evalSize, train_image, train_label, model):
    train_image_batch = []
    train_label_batch = []
    batch_indices = np.random.choice(len(train_image), evalSize)
    for x in batch_indices: 
        train_image_batch.append(train_image[x])
        train_label_batch.append(train_label[x])
    train_image_batch = np.array(train_image_batch)
    train_label_batch = np.array(train_label_batch)
    train_image_batch = train_image_batch.reshape(evalSize, 28, 28, 1)

    score = model.evaluate(train_image_batch, train_label_batch, verbose=0)
    return score

def weightsInfo(weights):
    layers = len(weights)
    totalMean = 0
    totalVar = 0
    for row in weights:
        totalMean += np.mean(row)
        totalVar += np.var(row)
    return totalMean/layers, totalVar/layers

if __name__ == "__main__":
    # print(tau_setup_all)
    # model import, time generation, data organisation
    [use_fixed_averaging_slots, train_image, train_label, test_image, test_label, train_label_orig, indices_each_node_case] = setup()

    # clients joining the session
    client_sock_all = socketsInFLSystem()

    # begin simulation
    stat = simulationBegin(single_run)

    allGlobalLosses = []

    tau_setup_all = [[1],[3],[6],[10],[15],[20], [30], [40], [60], [70],[100]]

    control_alg = None
   
    for sim in sim_runs:

        if batch_size >= total_data:  # Read data again for different sim. round
            train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

            # This function takes a long time to complete,
            indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

        for case in case_range:

            for tau_setup in tau_setup_all:
                tau_setup = tau_setup[0]
                print("####################################")
                print("############### %d ###############" % (tau_setup))
                print("####################################")

                # initalise w_global
                model = getCNNModel(step_size)
                w_global = model.get_weights()
                dim_w = np.array(w_global).shape
                model.summary()

                # setup min loss tracking
                w_global_min_loss = None
                min_loss = np.inf

                # tau processing (adaptive or fixed)
                tau_config = tau_setup

                # initalising nodes
                for n in range(0, n_nodes):
                    indices_this_node = indices_each_node_case[case][n]
                    msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset,
                        num_iterations_with_same_minibatch_for_tau_equals_one, step_size,
                        batch_size, total_data, None, indices_this_node, read_all_data_for_stochastic,
                        use_min_loss, sim]
                    send_msg(client_sock_all[n], msg)

                # Wait until all clients complete data preparation and sends a message back to the server
                for n in range(0, n_nodes):
                    recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')
                print('All clients connected')

                (time_global_aggregation_all, total_time, total_time_recomputed,
                    it_each_local, it_each_global, is_last_round, is_eval_only,
                    tau_new_resume) = variablesSetup()

                # Loop for multiple rounds of local iterations + global aggregation
                currentEpoch = 0
                sumLoss = 0
                sumAcc = 0

                # learning process
                print('Start learning')
                while True:

                    print('---------------------------------------------------------------------------')
                    currentEpoch += 1
                    print('Current Epoch: ', currentEpoch ,', Tau config:', tau_config)

                    time_total_all_start = time.time()

                    # send current weights and tau
                    if currentEpoch != 1:
                        w_global = w_global_min_loss
                    for n in range(0, n_nodes):
                        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, True]
                        send_msg(client_sock_all[n], msg)

                    w_global = np.zeros(dim_w)

                    data_size_total = 0
                    time_all_local_all = 0
                    data_size_local_all = []
                    tau_actual = 0

                    # recieve updated weights
                    for n in range(0, n_nodes):
                        msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                        # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                        # loss_last_global, loss_w_prev_min_loss]
                        w_local = msg[1]
                        time_all_local = msg[2]
                        tau_actual = max(tau_actual, msg[3])  # Take max of tau because we wait for the slowest node
                        data_size_local = msg[4]

                        w_local = np.array(w_local)

                        w_global = w_global + w_local * data_size_local
                        data_size_local_all.append(data_size_local)
                        data_size_total += data_size_local
                        time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node

                    w_global /= data_size_total
                    for arr in w_global:
                        if np.sum(arr) == np.NaN:
                            print('*** w_global is NaN, using previous value')
                            w_global = w_global_min_loss   # If current w_global contains NaN value, use previous w_global
                            use_w_global_prev_due_to_nan = True
                            break
                    else:
                        use_w_global_prev_due_to_nan = False

                    print("Minimum loss: " + str(min_loss))

                    # If use_w_global_prev_due_to_nan, then use tau = 1 for next round
                    if not use_w_global_prev_due_to_nan:
                        w_global_prev = w_global
                        # comparing current global weights to 
                        model.set_weights(w_global_prev)
                        score = evalModel(500, train_image, train_label, model) # batch size of 500 for global eval. - need reason for this
                        if currentEpoch > 4:
                            sumLoss += score[0]
                            sumAcc += score[1]
                        if score[0] < min_loss:
                            print("Updating Min Loss")
                            w_global_min_loss = w_global_prev
                            min_loss = score[0]
                        else: print("Using current min loss")
                        
                        if tau_new_resume is not None:
                            tau_new = tau_new_resume
                            tau_new_resume = None
                        else:
                            tau_new = tau_config
                    else:
                        if tau_new_resume is None:
                            tau_new_resume = tau_config
                        tau_new = 1

                    # Calculate time
                    time_total_all_end = time.time()
                    time_total_all = time_total_all_end - time_total_all_start
                    time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                    # print('Time for one local iteration:', time_all_local_all / tau_actual)
                    # print('Time for global averaging:', time_global_aggregation_all)

                    if use_fixed_averaging_slots:
                        if isinstance(time_gen, (list,)):
                            t_g = time_gen[case]
                        else:
                            t_g = time_gen
                        it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_actual)) / tau_actual)
                        it_each_global = t_g.get_global(1)[0]
                    else:
                        it_each_local = max(0.00000001, time_all_local_all / tau_actual)
                        it_each_global = time_global_aggregation_all

                    #Compute number of iterations is current slot
                    total_time_recomputed += it_each_local * tau_actual + it_each_global

                    #Compute time in current slot
                    total_time += time_total_all

                    # Check remaining resource budget (use a smaller tau if the remaining time is not sufficient)
                    is_last_round_tmp = False

                    if use_min_loss:
                        tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * (tau_new + 1) + it_each_global * 2
                    else:
                        tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * tau_new + it_each_global

                    if tmp_time_for_executing_remaining < max_time:
                        tau_config = tau_new
                    else:
                        if use_min_loss:  # Take into account the additional communication round in the end
                            tau_config = int((max_time - total_time_recomputed - 2 * it_each_global - it_each_local) / it_each_local)
                        else:
                            tau_config = int((max_time - total_time_recomputed - it_each_global) / it_each_local)

                        if tau_config < 1:
                            tau_config = 1
                        elif tau_config > tau_new:
                            tau_config = tau_new

                        is_last_round_tmp = True

                    if is_last_round:
                        break

                    if is_eval_only:
                        tau_config = 1
                        is_last_round = True

                    if is_last_round_tmp:
                        if use_min_loss:
                            is_eval_only = True
                        else:
                            is_last_round = True

                model.set_weights(w_global_min_loss)
                score = evalModel(10000, train_image, train_label, model)
                avgLoss = sumLoss/(currentEpoch-4)
                avgAcc = sumAcc/(currentEpoch-4)

                allGlobalLosses.append((tau_setup, score[1], score[0], avgLoss, avgAcc))

        print("all Global Losses", allGlobalLosses)
        accuracyMaxList = []
        accAvgList = []
        lossAvgList = []
        for (_,maxAcc,_,avgLoss,avgAcc) in allGlobalLosses: 
            accuracyMaxList.append(maxAcc)
            accAvgList.append(avgAcc)
            lossAvgList.append(avgLoss)
        plt.plot(accuracyMaxList)
        plt.title("Max Accuracy for each  model")
        plt.show()
        plt.plot(accAvgList)
        plt.title("Average Accuracy for each  model")
        plt.show()
        plt.plot(lossAvgList)
        plt.title("Average Loss for each  model")
        plt.show()
