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
from connectFour import historyToImage

config = cfg()


# placeholder
# will return (p,v)
# policy vector and expected value of the state
def random_network():
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

class MCTS():
    # compute (Q+U)
    def compute_action_vals(self, node):
        q_plus_u = numpy.array([(child.Q() + child.U()) for (action,child) in node.children.items()])
        return q_plus_u

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
    def add_exploration_noise(self,node):
        actions = node.children.keys()
        noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
        frac = config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior_prob = node.children[a].prior_prob * (1 - frac) + n * frac
        return node 

    def run_sim(self, game,network, useNetwork):
        T = 10
        root = GameNode({},1)
        root.visit_count = 1
        root.player = game.player()
        if useNetwork:
            cuda = torch.device('cuda')
            # evaluate network for this state
            img = historyToImage(game.history, game.width, game.height)
            (timeSteps, w,h) = img.shape
            diff = T-timeSteps
            if diff > 0:
                img = torch.cat([img, torch.zeros(diff,w,h,device=cuda)])
            elif diff < 0:
                img = img[timeSteps-T-1:timeSteps-1, :,:]
            img=img.unsqueeze(0)
            #print(img)
            log_policy,val = network(img)
            policy = torch.exp(log_policy)
            #print(f"p: {policy}, val: {val}")
        else:
            policy,val = random_network()

        #initialize the children nodes
        for i,p in enumerate(policy):
            root.children[i] = GameNode(root, p)

        root = self.add_exploration_noise(root)
        for _ in range(200):
        #for _ in range(config.num_simulations):
            node = root
            trial = game.clone()

            #traverse down the tree
            while not node.isLeaf() and not trial.gameWinner() and not trial.gameTie():
                #select the action maximizing expected value                
                action_vals = self.compute_action_vals(node)

                #take the action
                success = False
                n_fail = 0
                while(not success):
                    act = numpy.argmax(action_vals)
                    if act.dtype != numpy.int64:
                        act = act.item()
                    success = trial.move(act)
                    if not success:
                        action_vals[act] = -numpy.inf
                    if n_fail > 5: 
                        print("FAIL")
                        import pdb
                        pdb.set_trace()
                        print(self.compute_action_vals(node))
                    n_fail += 1

                node = node.children[act]
                node.player = trial.player()
    
            if(useNetwork):
                img = historyToImage(game.history, game.width, game.height)
                (timeSteps, w,h) = img.shape
                diff = T-timeSteps
                if diff > 0:
                    img = torch.cat([img, torch.zeros(diff,w,h,device=cuda)])
                elif diff < 0:
                    img = img[timeSteps-T-1:timeSteps-1, :,:]

                img = img.unsqueeze(0)
                log_policy,val = network(img)
                policy = torch.exp(log_policy)
            else:
                policy,val = random_network()
            for i,p in enumerate(policy):
                node.children[i] = GameNode(node, p)

            root = self.update_tree(node, val, trial.player())

        return root

    # choose an action "proportional for exploration
    # or greedily for exploitation wrt visit count"
    def step(self,root,num_actions):
        v = [(a,root.children[a].visit_count) if a in root.children else (a,0)
            for a in range(num_actions)]
        v.sort(key=lambda x: x[0] )
        visits = torch.tensor([v for (a,v) in v]).float()
        probs = functional.softmax(visits)
        return probs
            

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

def renormalize(dist, idx):
    dist[idx] = 0
    total = sum(dist)
    dist = dist/total
    return dist

def play_game(game,network, useNetwork):
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
        if useNetwork and game.turnNum >= 10:
            root = mcts.run_sim(game,network,True)
        else:
            root = mcts.run_sim(game,network,False)
        
        action_probs = mcts.step(root, game.width)
        # choose action according to distribution,
        # correcting if selecting illegal move
        success = False
        moves = list(range(game.width))
        while(not success):
            action = random.choices(moves,action_probs)[0]  
            success = game.move(action)
            if not success:
                action_probs = renormalize(action_probs, action)
        #if useNetwork:
        #    print(f"action {action}")
        
        images.append(historyToImage(game.history, game.width, game.height))
        policies.append(root.computeTargetPolicy(game.width)) 
        #print(f"moved: {game.state()}")

    #update the value once the game is over
    targets = []
    if winner == 1:
        playerToReward = {1:1, 2:-1}
    elif winner == 2:
        playerToReward = {1:-1, 2:1}
    else:
        playerToReward = {1:0, 2:0}
    for i,p in enumerate(policies):
        targets.append((p,playerToReward[i%2 + 1]))

    return  images, targets
#shutil.rmtree("testMCTS")
#game = ConnectFour(5,5,True,"testMCTS")
#play_game(game)
#print(game.history)
#watchGame("testMCTS")
