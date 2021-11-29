import numpy as np
from scipy.stats import mode
import copy
from TreeClassifier import TreeClassifier
from SimulatedAnnealing import SimulatedAnnealing
import matplotlib.pyplot as plt
import time
import argparse
import sys
import hashlib


TICTACTOE = 1
SYNTHETIC = 0

DATASET_TXT = {
    TICTACTOE: "Tic-tac-toe",
    SYNTHETIC: "Synthetic" 
}


# LINEAR_SCHED = 0

# SCALING = {
    
# }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=int, help="dataset to use (0=synthetic, 1=tic-tac-toe)", default=0)
    parser.add_argument('--itrs', '-i', type=int, help="number of iterations to run", default=1000)
    parser.add_argument('--max_depth', '-m', type=int, help="maximum depth of tree (1 corresponds to just a root, no splits)", default=3)
    parser.add_argument('--leaf_penalty', '-l', type=int, help="penalty to objective per leaf in tree", default=0.01)
    parser.add_argument('--initial_temp', '-t', type=int, help="Starting temperature", default=1)
    parser.add_argument('--prob_contract', '-c', type=int, help="probability to contract a split when proposing a new tree", default=0.05)
    parser.add_argument('--prob_split', '-s', type=int, help="probability to create a split when proposing a new tree", default=0.1)
    
    parsed_args = parser.parse_args()
    dataset = parsed_args.data

    hyperparams = {}

    num_its = parsed_args.itrs#1000#100000 #2000
    hyperparams["itrs"] = parsed_args.itrs

    check_time_after = parsed_args.itrs

    leaf_penalty = parsed_args.leaf_penalty
    hyperparams["leaf_penalty"] = parsed_args.leaf_penalty

    depth_budget = parsed_args.max_depth
    hyperparams["max_depth"] = parsed_args.max_depth

    hyperparams["temp"] = parsed_args.initial_temp
    hyperparams["contract"] = parsed_args.prob_contract
    hyperparams["split"] = parsed_args.prob_split
    hyperparams["scaling"] = "linear" #just use linear scaling for now
    hash = hashlib.md5(str(hyperparams).encode('utf8')).hexdigest()[:6]

    if dataset == SYNTHETIC: 
        data = np.array([[True,  True, True,  True,  0], 
                    [True,  True, True,  True,  1], 
                    [True,  True, True,  False, 0], 
                    [True,  True, False, True,  0], 
                    [True,  True, False, False, 1]])
    elif dataset == TICTACTOE: 
        data = np.loadtxt('dataset/tic-tac-toe.txt')
        #adjust because the first column is labels for this dataset: 
        #try np.roll?
        data = np.concatenate((data[:,1:],data[:, 0:1]), axis = 1)
    else: 
        print("dataset not recognized - must be 0 or 1")

    sim_an = SimulatedAnnealing(data, depth_budget = depth_budget, seed = 1, leaf_penalty=leaf_penalty, start_temp=hyperparams["temp"], prob_split = hyperparams["split"], prob_contract = hyperparams["contract"])
    #add acc for initial state: 
    objectives = [sim_an.current_tree.objective(leaf_penalty)]
    start_time = time.perf_counter()
    times = [0] #take times at different intermediate points
    for i in range(int(num_its/check_time_after)): 
        curr_objs, cur_tree = sim_an.iterate(check_time_after)
        objectives = objectives + curr_objs
        times.append(time.perf_counter() - start_time) #may throw off time slightly to take this measure every time (but also not sure if it's safe to assume time is completely linear with iteration count)
    end_time = time.perf_counter()
    print("total time: " + str(end_time - start_time))
    print("final objective: " + str(objectives[-1]))
    print("final tree: ")
    sim_an.current_tree.print(show_data=False)

    #plot vs itrs
    plt.plot(np.arange(0, len(objectives)), objectives) #may want to compress to change iteration scale so that we can use fewer 0's
    plt.xticks(np.arange(0, len(objectives), 10000)) #could also show exp notation
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Objective vs Iteration for Simulated Annealing \n (Where Objective = Training Error + 0.01*{# leaves}')# \n on a Simple Dataset (4 Features, 5 Examples)')
    #show hyperparams on slide in small font, discuss how simple it is?
    plt.savefig('figs/anneal_itrs_' + DATASET_TXT[dataset] + '_' + str(hash) + '.png')
    plt.show()

    sys.stdout = open('logs/anneal_' + DATASET_TXT[dataset] + '_' + str(hash) + '.txt', 'w')
    print("hyperparams: ")
    print(str(hyperparams))
    print("total time: " + str(end_time - start_time))
    print("final objective: " + str(objectives[-1]))
    print("final tree: ")
    sim_an.current_tree.print(show_data=False)
    sys.stdout.close()
    # with open('logs/anneal_' + DATASET_TXT[dataset] + ".txt", "w") as outlog: 
    #     outlog.write("total time: " + str(end_time - start_time))
    #     print("final objective: " + str(objectives[-1]))
    #     print("final tree: ")
    #     sim_an.current_tree.print(show_data=False)

    #plot vs time: 
    # plt.plot(times, np.array(objectives)[np.arange(0, len(objectives), check_time_after)])
    # plt.savefig('figs/acc_vs_time.png')
    # plt.xlabel('Time(s)')
    # plt.xticks(np.arange(0, end_time - start_time, 10))
    # plt.ylabel('Objective')
    # plt.title('Objective vs Time(s) for Simulated Annealing \n (Where Objective = Training Error + 0.01*{# leaves} ')#\n on a Simple Dataset (4 Features, 5 Examples)')
    # plt.savefig('figs/anneal_time_' + DATASET_TXT[dataset] + '.png')
    # plt.show()

