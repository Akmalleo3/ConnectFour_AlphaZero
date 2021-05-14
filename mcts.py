import math
import numpy
from numpy import random as numpy_random

import os
import random
import shutil
import torch
import time
from torch.nn import functional 

from AlphaZeroConfig import AlphaZeroConfig as cfg
from connectFour import ConnectFour
from connectFour import historyToImage

config = cfg()


# placeholder
# will return (p,v)
# policy vector and expected value of the state
def random_network(width):
    n_legal_moves = width
    x = torch.rand(1, n_legal_moves)
    p = functional.softmax(x,dim=1)
    p = p.squeeze()
    v = torch.rand(1)
    return p,v

class GameNode():
    def __init__(self,parent, prior):
        self.visit_count = 0
        self.total_action_value = 0
        self.prior_prob = prior
        self.children = {}
        self.parent = parent
        self.player = 0

    #mean action value
    def Q(self):
        if self.visit_count == 0:
            return 0
        return self.total_action_value/self.visit_count

    def U(self):
        c = math.log((1+self.parent.visit_count + config.pb_c_base)/config.pb_c_base)
        c = c + config.pb_c_init
        # N(s) = parent visit count, N(s,a) is this node visit count
        u = c*self.prior_prob*math.sqrt(self.parent.visit_count)
        u = u/(1 + self.visit_count)
        return u

    def print(self):
        print(f"(N = {self.visit_count} W = {self.total_action_value} Q = {self.Q()} P = {self.prior_prob} )")
        
    def isLeaf(self):
        return len(self.children) == 0

    def computeTargetPolicy(self, num_actions):
        totalChildVisits = sum(child.visit_count for child in self.children.values())
        childVisits = [self.children[a].visit_count if a in self.children else 0
            for a in range(num_actions)]

        policy = [vc/totalChildVisits for vc in childVisits]
        return policy 

# set probability at idx = 0, and recompute probabilities
def renormalize(dist, idx):
    dist[idx] = 0
    total = sum(dist)
    dist = dist/total
    return dist


class MCTS():
    # compute (Q+U)
    def compute_action_vals(self, node):
        q_plus_u = numpy.array([(child.Q() + child.U()) for (action,child) in node.children.items()])
        return q_plus_u

    def select_best_action(self, node, game):
        action_vals = self.compute_action_vals(node)
        success = False
        while(not success):
            act = numpy.argmax(action_vals)
            if act.dtype != numpy.int64:
                act = act.item()
            success = game.move(act)
            if not success:
                action_vals[act] = -numpy.inf
        return act
    
    def compute_visit_probabilities(self,root,num_actions):
        v = [(a,root.children[a].visit_count) if a in root.children else (a,0)
            for a in range(num_actions)]
        v.sort(key=lambda x: x[0] )
        visits = torch.tensor([v for (a,v) in v]).float()
        probs = functional.softmax(visits)
        return probs
    

    # choose an action "proportional for exploration
    # or greedily for exploitation wrt visit count"
    def take_most_visited_action(self,root, game):
        action_probs = self.compute_visit_probabilities(root, game.width)
        # choose action according to distribution,
        # correcting if selecting illegal move
        success = False
        moves = list(range(game.width))
        while(not success):
            action = random.choices(moves,action_probs)[0]  
            success = game.move(action)
            if not success:
                action_probs = renormalize(action_probs, action)
        return game

    def print_tree(self,root):
        node = root
        node.print()
        if not node.isLeaf():
            for (_,child) in node.children.items():
                self.print_tree(child)
            
    # incorporate v from network
    # into value estimations for each
    # node along the path. increment visit counts
    def update_tree(self, node,val, player):
        while 1:
            if node.player == player:
                node.total_action_value += val
            else:
                node.total_action_value += -val
            node.visit_count += 1
            if node.parent == {}:
                break
            node = node.parent

        return node

    # adjust prior probabilities of actions to encourage exploration
    def add_exploration_noise(self,node):
        actions = node.children.keys()
        noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior_prob = node.children[a].prior_prob * (1 - frac) + n * frac
        return node 

    # Evaluate the network at this state to get policy distribution
    # and initialize child node with these probabilities
    def expand_node(self, node, game, network, useNetwork,T):
        if useNetwork:
            img = historyToImage(game.history, game.width, game.height,T)
            log_policy,val = network(img)
            policy = torch.exp(log_policy)
        else:
            policy,val = random_network(game.width)

        #initialize the children nodes
        for i,p in enumerate(policy):
            node.children[i] = GameNode(node, p)
        return node, val
    
    def random_move(self,game):
        rng = numpy_random.default_rng()
        moved = False
        actions = game.legalMoves()
        while(not moved):
            move = rng.integers(len(actions))
            moved = game.move(actions[move])
        return game, move

    def random_playout(self, node,game):
        while not game.gameWinner() and not game.gameTie():
            game, move = self.random_move(game)
            prob = 1/game.width
            for i in range(game.width):
                node.children[i] = GameNode(node, prob)
            node = node.children[move]
            node.player = game.player()
        return game, node

    def run_sim(self, game,network, useNetwork,T):
        root = GameNode({},1)
        root.visit_count = 1
        root.player = game.player()
        root, val = self.expand_node(root, game, network, useNetwork,T)

        root = self.add_exploration_noise(root)
        for _ in range(50):
            node = root
            trial = game.clone()

            #traverse down the tree
            while not node.isLeaf() and not trial.gameWinner() and not trial.gameTie():
                #select the action maximizing expected value
                act = self.select_best_action(node, trial)
                node = node.children[act]
                node.player = trial.player()

            # Once at a leaf, evaluate the game using the network
            # or complete one random playout from C to the end
            if useNetwork:
                node, val = self.expand_node(node, trial, network, useNetwork,T)
                player = trial.player()
            else:
                trial, node = self.random_playout(node, trial)
                player = trial.gameWinner()
                if player:
                    val = 1
                else:
                    val = 0

            # Use the results of the playout to update the tree value estimations/visit counts
            root = self.update_tree(node, val, player)

        return root

    

def watchGame(gameDir):
    #os.system('clear')
    turns = os.listdir(gameDir)
    turns.sort()
    frames = []
    for turnFile in turns:
        with open(f"{gameDir}/{turnFile}", "r") as f:
            frames.append(f.readlines())
    for frame in frames:
        print("".join(frame))
        time.sleep(2)
        os.system('clear')



def play_game(network, useNetwork,T, width, height):
    game = ConnectFour(width,height,True)

    mcts = MCTS()
    policies = []
    images = []
    while True:
        if game.turnNum > game.width*game.height:
            break
        winner = game.gameWinner()
        if(winner):
            break
        if game.gameTie():
            break
        useNetwork = useNetwork and game.turnNum >= T

        root = mcts.run_sim(game,network,useNetwork,T)
        game = mcts.take_most_visited_action(root,game)

        # Store the state and the observed action distribution for training
        images.append(historyToImage(game.history, game.width, game.height,T))
        policies.append(root.computeTargetPolicy(game.width)) 

    #update the target value once the game is over and outcome is known
    targets = []
    if winner == 1:
        playerToReward = {1:1, 2:-1} # player 1 wins, player two loses
    elif winner == 2:
        playerToReward = {1:-1, 2:1} # player 2 wins, player one loses
    else:
        playerToReward = {1:0, 2:0} #tie 
    for i,p in enumerate(policies):
        targets.append((p,playerToReward[i%2 + 1]))

    return  images, targets

