import math
import numpy
import os
import random
import shutil
import torch
import time
from torch.nn import functional 

from AlphaZeroConfig import AlphaZeroConfig as cfg
from connectFour import ConnectFour

config = cfg()


# placeholder
# will return (p,v)
# policy vector and expected value of the state
def network():
    n_legal_moves = 5
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
        c = math.log((1+self.visit_count + config.pb_c_base)/config.pb_c_base)
        c = c + config.pb_c_init
        # N(s) = parent visit count, N(s,a) is this node visit count
        u = c*self.prior_prob*math.sqrt(self.parent.visit_count)
        u = u/(1 + self.visit_count)
        return u

    def print(self):
        print(f"(N = {self.visit_count} W = {self.total_action_value} Q = {self.Q()} P = {self.prior_prob} )")
        

    def isLeaf(self):
        return len(self.children) == 0

class MCTS():
    #select max_a (Q+U)
    def select_action(self, node):
        q_plus_u = numpy.array([(child.Q() + child.U()) for (action,child) in node.children.items()])
        return numpy.argmax(q_plus_u)

    def print_tree(self,root):
        node = root
        node.print()
        if not node.isLeaf():
            for (action,child) in node.children.items():
                self.print_tree(child)
            


    # incorporate v from network
    # into value estimations for each
    # node along the path. increment visit counts
    def update_tree(self, node,val, player):
        while 1:
            if node.player == player:
                node.total_action_value += val
            else:
                node.total_action_value += (1-val)
            node.visit_count += 1
            if node.parent == {}:
                break
            node = node.parent

        return node

    def run_sim(self, game):
        root = GameNode({},1)
        root.player = game.player()
        # evaluate network for this state
        s = game.state()
        policy,val = network()

        #initialize the children nodes
        for i,p in enumerate(policy):
            root.children[i] = GameNode(root, p)

        for _ in range(5):
        #for _ in range(config.num_simulations):
            node = root
            trial = game.clone()

            #traverse down the tree
            while not node.isLeaf():
                s = trial.state()
                #select the action maximizing expected value
                actions = trial.legalMoves()
                act = self.select_action(node)
                #take the action
                trial.move(act)
                node = node.children[act]
                node.player = trial.player()

            policy,val = network()
            for i,p in enumerate(policy):
                node.children[i] = GameNode(node, p)

            root = self.update_tree(node, val, trial.player())

        return root

    # choose an action "proportional for exploration
    # or greedily for exploitation wrt visit count"
    def step(self,root):
        v = [(action,child.visit_count) for (action, child) in root.children.items()]
        v.sort(key=lambda x: x[0] )
        visits = torch.tensor([v for (a,v) in v]).float()
        softmax = functional.softmax(visits)
        #TODO replace with random.choices 
        #sample 
        i  = random.uniform(0,1)
        tot = 0
        for idx,x in enumerate(softmax):
            tot = tot + x
            if i <= tot:
                return idx
            

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


def play_game(game):
    mcts = MCTS()
    #print("init state")
    game.state()
    while True:
        if game.turnNum > game.width*game.height:
            break
        winner = game.gameWinner()
        if(winner):
            break
        if game.gameTie():
            break
        root = mcts.run_sim(game)
        #print(f"state before: {game.state()}")
        #print("tree")
        #mcts.print_tree(root)
        action = mcts.step(root)
        #print(f"Action: {action}")
        game.move(action)
        #print(f"moved: {game.state()}")

#shutil.rmtree("testMCTS")
#game = ConnectFour(5,5,True, "testMCTS")
#play_game(game)
#print(game.history)
#watchGame("testMCTS")
