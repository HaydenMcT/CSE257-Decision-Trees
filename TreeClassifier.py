import numpy as np
from scipy.stats import mode
import copy

class TreeClassifier: #tree classifier of specified structure
    def __init__(self):
        self.terminal = True

    def split(self, rule, rule_name = '', left=None, right=None): #left is the tree to use for rule evaluating to false, right is for true
        self.terminal = False
        self.rule = rule
        self.rule_name = rule_name
        self.left = left if left is not None else TreeClassifier()
        self.right = right if right is not None else TreeClassifier()

    # input: path: a list of strings, either "left" or "right", to take from the root to find a leaf 
    #        (stopping as soon as a leaf is reached, even if more entries in the path exist)
    # finds the leaf corresponding to a given path, and contracts it
    def contract_leaf(self, path): 
        if self.terminal: 
            # print("contraction impossible for leaves of this subtree - already a leaf")
            return
        cur_node = self
        for branch in path: 
            parent_node = cur_node
            cur_node = self.left if branch is "left" else self.right
            if cur_node.terminal: 
                break
        # if not cur_node.terminal: 
            # print("contracting a node that is still a parent... path ended before leaf found!")
        parent_node.terminal = True
        return

    def find_node(self, path): #recursive
        cur_node = self
        for branch in path: 
            if cur_node.terminal: 
                # print("path too long! returning leaf which matches the first part of path")
                break
            cur_node = cur_node.left if branch is "left" else cur_node.right
        return cur_node

    def split_leaf(self, path, rule, rule_name = '', left=None, right = None): 
        to_split = self.find_node(path)
        if not to_split.terminal: 
            print("splitting a non-terminal node... path ended before leaf found")
        to_split.split(rule, rule_name, left, right)

    def get_leaf_paths(self): 
        if self.terminal: 
            return [[]]
        
        leaves = []
        for leaf in self.left.get_leaf_paths(): 
            leaves.append(["left"] + (leaf))
        for leaf in self.right.get_leaf_paths(): 
            leaves.append(["right"]+(leaf))
        #print(leaves)
        return leaves

    
    def build(self, train): 
        #data has its last column as the labels (0 or 1)
        self.train_data = train
        if not self.terminal:
            if train.size == 0: 
                self.left.build(train)
                self.right.build(train) 
            else: 
                true_idcs = np.where(np.apply_along_axis(self.rule, 1, train))[0]
                left_data = np.delete(train, true_idcs, axis=0)
                right_data = train[true_idcs, :]
                self.left.build(left_data)
                self.right.build(right_data)

    def predict_one(self, prediction_point): #prediction_point is just one row, not a dataset
        if self.terminal: 
            if self.train_data.size == 0: 
                return 0 #with no data predict 0
            else: 
                return mode(self.train_data[:,-1], axis=None)[0][0] #ASSUMES train_data has its last column as the labels (0 or 1)
        else: 
            if self.rule(prediction_point):
                return self.right.predict_one(prediction_point)
            else: 
                return self.left.predict_one(prediction_point)

    def predict(self, prediction_points): #prediction_points are all rows
        predictions = np.zeros(prediction_points.shape[0])
        for idx, prediction_point in enumerate(prediction_points): 
            predictions[idx] = self.predict_one(prediction_point) #todo: improve efficiency
        return predictions


    def objective(self, leaf_penalty): 
        return (1 - np.mean(self.predict(self.train_data) == self.train_data[:, -1])) + leaf_penalty * self.num_leaves()
    
    def height(self): 
        if self.terminal: 
            return 1
        else: 
            return 1 + max(self.left.height(), self.right.height())

    def num_leaves(self): 
        if self.terminal: 
            return 1
        else: 
            return self.left.num_leaves() + self.right.num_leaves()

    def print(self, indent = 0, show_data=False):
        printable_indent = ' '*indent
        if self.terminal: 
            print(printable_indent + "predict " + str(self.predict_one([])))
            if show_data: 
                print(printable_indent + "  ^ based on data: " + str(self.train_data) )
                # print(self.train_data.size)
                # # if(self.train_data.size == 0): 
                # #     print("no data")
                # #     print(self.predict_one([1, 2, 3, 4, 0]))
                # print(mode(self.train_data[:,-1], axis=None)[0])
                
        else: 
            print(printable_indent + "if "+ self.rule_name +" is true: " ) #todo: turn rule into more descriptive string
            self.right.print(indent+2, show_data)
            print(printable_indent + "else: ")
            self.left.print(indent + 2, show_data)
