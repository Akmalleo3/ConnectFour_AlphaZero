
import torch
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

    def isLeaf(self):
        return len(self.children) > 0

class MCTS():
    #select max_a (Q+U)
    def select_action(self, actions, node):
        q_plus_u = [child.Q() + child.U() for child in node.children]
        return argmax(q_plus_u)

    # incorporate v from network
    # into value estimations for each
    # node along the path. increment visit counts
    def update_tree(self, node,val, player):
        while node.parent != {}:
            if node.player == player:
                node.total_action_val += val
            else:
                node.total_action_value += (1-val)
            node.visit_count += 1
            node = node.parent

    def run_sim(self, game):
        root = GameNode({},1)
        # evaluate network for this state
        s = game.state()
        policy,val = network()

        #initialize the children nodes
        for i,p in enumerate(policy):
            root.children[i] = GameNode(root, p)
            root.children[i].player = game.player()

        for _ in range(config.num_simulations):
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

            policy,val = network()
            for i,p in enumerate(policy):
                node.children[i] = GameNode(node, p)
                node.children[i].player = trial.player()

            self.update_tree(node, val, trial.player())
        return root

    # choose an action "proportional for exploration
    # or greedily for exploitation wrt visit count"
    def step(self,root):
        v = [(action,child.visit_count) for (action, child) in root.children.items()]
        v.sort(key=lambda x: x[0] )
        print(v)
        visits = torch.tensor([v for (a,v) in v])
        print(visits)
        softmax = functional.softmax(visits)
        #sample 
        i  = random.uniform(0,1)
        for idx,x in enumerate(softmax):
            if i <= x:
                return idx
            


def play_game():
    mcts = MCTS()
    game = ConnectFour(4,4,1)
    while True:
        winner = game.gameWinner()
        if(winner):
            break
        if game.gameTie():
            break
        root = mcts.run_sim(game)
        print(root.children)
        action = mcts.step(root)
        game.move(action)

play_game()

