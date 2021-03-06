from TreeClassifier import TreeClassifier
import random
import numpy as np
import copy

#minimize training error with depth constraint
class SimulatedAnnealing: 
    # assume data is formatted with each feature as a column, and the labels as the last column
    # binary features for now
    def __init__(self, train_data, depth_budget = 3, seed=None, leaf_penalty = 0.01, start_temp = 1, prob_split = 0.1, prob_contract = 0.05):
        self.leaf_penalty = leaf_penalty
        self.current_tree = TreeClassifier() 
        self.current_tree.build(copy.deepcopy(train_data))
        self.train_data = train_data
        self.start_temp = start_temp
        self.n = 1
        self.temp = lambda: self.start_temp/self.n #lambda: self.start_temp * 0.9**self.n #lambda: self.start_temp*np.log(2)/np.log(self.n+1) #lambda: self.start_temp * 0.99**self.n
        self.prob_split = prob_split
        self.prob_contract = prob_contract
        self.depth_budget = depth_budget
        self.rules = []
        self.rule_names = {}
        for i in range(train_data.shape[1] - 1): 
            rule = lambda row, i=i: row[i] #i=i for reasons discussed in https://stackoverflow.com/questions/938429
            self.rules.append(rule)
            self.rule_names[rule] = "feature " + str(i)
        if seed is not None: 
            random.seed(seed)
            np.random.seed(seed)

    # modifies current tree...
    def propose_move(self):
        proposal_tree = copy.deepcopy(self.current_tree)
        #make new tree based on current tree
        leaf_paths = proposal_tree.get_leaf_paths()
        random.shuffle(leaf_paths)
        for leaf_path in leaf_paths:
            decide_to_change = random.random()
            if decide_to_change < self.prob_split and len(leaf_path) < self.depth_budget - 1: 
                rule_idx = np.random.randint(len(self.rules))
                rule_to_split_on = self.rules[rule_idx]
                proposal_tree.split_leaf(leaf_path, rule_to_split_on, rule_name = self.rule_names[rule_to_split_on])
                break
            elif decide_to_change > self.prob_split and decide_to_change < self.prob_contract + self.prob_split: 
                proposal_tree.contract_leaf(leaf_path) #could be terminal, in which case nothing should happen
                break
        proposal_tree.build(copy.deepcopy(self.train_data))
        return proposal_tree

        # alternative proposal distribution: 
        # proposal_tree = copy.deepcopy(self.current_tree)
        # #make new tree based on current tree
        # for leaf_path in proposal_tree.get_leaf_paths(): #double check this doesn't have weird behaviour since we're modifying the tree
        #     decide_to_change = random.random()
        #     if decide_to_change < self.prob_split and len(leaf_path) < self.depth_budget - 1: 
        #         rule_idx = np.random.randint(len(self.rules))
        #         rule_to_split_on = self.rules[rule_idx]
        #         proposal_tree.split_leaf(leaf_path, rule_to_split_on, rule_name = self.rule_names[rule_to_split_on])
        #     elif decide_to_change > self.prob_split and decide_to_change < self.prob_contract + self.prob_split: 
        #         proposal_tree.contract_leaf(leaf_path) #check for terminal?
        # proposal_tree.build(copy.deepcopy(self.train_data))
        # return proposal_tree


    #either accept a move or reject, based on objective and temperature
    def accept_or_reject(self, proposed_tree): 
        obj_decrease = self.current_tree.objective(self.leaf_penalty) - proposed_tree.objective(self.leaf_penalty)
        if obj_decrease >= 0: 
            self.current_tree = proposed_tree
        else:
            prob_accept = np.exp(obj_decrease/self.temp())
            if random.random() < prob_accept: 
                self.current_tree = proposed_tree

    def iterate(self, num_its=1, seed=None):
        if seed is not None: 
            random.seed(seed)
            np.random.seed(seed)
        obj_over_time = []
        for i in range(num_its):
            self.accept_or_reject(self.propose_move())
            self.n += 1
            obj_over_time.append(self.current_tree.objective(self.leaf_penalty))
            if self.n % 1000 == 0: #for longer runs, give intermediate updates
                print("at iter " + str(self.n) + ": objective == " + str(self.current_tree.objective(self.leaf_penalty)))
        return obj_over_time, self.current_tree
        

