from policyValueNet import PolicyValueNet
import torch 
from connectFour import ConnectFour, historyToImage
from numpy import random as numpy_random
import random 
import os
from mcts import MCTS, renormalize
import time

cuda = torch.device('cuda')

def watchGame(gameDir):
    os.system('clear')
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
#watchGame("testGame1")
#watchGame("exampleWin1")


def model_move(model,game, T):
    img = historyToImage(game.history, game.width, game.height,T)

    log_policy,val = model(img)
    #print(img)
    policy = torch.exp(log_policy).cpu().detach().numpy()
    moved = False
    while(not moved):
        act = random.choices(list(range(len(policy))),policy)[0]
        moved = game.move(act)
        if not moved:
            policy = renormalize(policy,act)
    print(f"policy: {policy}")
    print(f"val: {val}")
    print(f"ModelMove: {act}")

    return game

def random_move(game):
    rng = numpy_random.default_rng()
    moved = False
    actions = game.legalMoves()
    while(not moved):
        move = rng.integers(len(actions))
        moved = game.move(actions[move])
    return game


def mcts_move(game, T):
    mcts = MCTS()

    root = mcts.run_sim(game,None,False,T)
    game = mcts.take_most_visited_action(root,game)
    #root.print()
    #for _,child in root.children.items():
    #    child.print()
        
    return game


def network_v_random(network,T, width, height,doDraw=False):
    game = ConnectFour(width,height,True,doDraw,"testGame1")
    while(not game.gameTie()):
        if game.player() == 2:
           game = random_move(game)
        elif game.player() == 1:
            game = model_move(network, game, T)
        if game.gameWinner():
            return game.gameWinner()

    return 0

def random_v_random(nothing, nobody,width,height):
    game = ConnectFour(width,height,True)
    while(not game.gameTie()):
        game = random_move(game)
        if game.gameWinner():
            return game.gameWinner()
    return 0

def random_v_mcts(nowork,T,width,height,doDraw=False):
    game = ConnectFour(width, height,True, doDraw, "testGame1")
    while not game.gameTie():
        if game.player() == 1:
            game = mcts_move(game,T)  
        else:
            game = random_move(game)
        if game.gameWinner():
            return game.gameWinner()
    return 0    

def network_v_mcts(model, T, width, height, doDraw=False):
    game = ConnectFour(width, height,True, doDraw, "testGame1")
    while not game.gameTie():
        if game.player() == 2:
            game = mcts_move(game,T)  
        else:
            game = model_move(model, game, T)
        if game.gameWinner():
            return game.gameWinner()
    return 0

def eval_network(player_scheme, network,T,width,height):
    n_trials = 500
    n_wins = 0
    n_draws = 0
    for _ in range(n_trials):
        winner = player_scheme(network,T,width,height)
        if winner == 1: 
            n_wins += 1
        elif winner == 0:
            n_draws += 1
    winrate = n_wins/n_trials 
    drawrate = n_draws/n_trials
    print(f"{winrate} winning rate of Player 1")
    print(f"{drawrate} draw rate")
    return winrate, drawrate


def eval_network_v_random(network=None,network_path="./testing.pth"):
    print("Player 1 trained net: ")
    print("Player 2 Random: ")
    T = 1
    device = 'cuda'
    if network == None:
        network = PolicyValueNet(7,6,2*T)
        network.load_state_dict(torch.load(network_path,map_location=device))
    network.cuda()
    network.eval()
    eval_network(network_v_random,network,T,7,6)


#eval_network_v_random()

def eval_network_v_mcts(network=None,network_path="./testing.pth"):
    print("Player 1 trained net: ")
    print("Player 2 MCTS: ")
    T = 1
    device = 'cuda'
    if network == None:
        network = PolicyValueNet(7,6,2*T)
        network.load_state_dict(torch.load(network_path,map_location=device))
    network.cuda()
    network.eval()
    eval_network(network_v_mcts,network,T,7,6)

#eval_network_v_mcts()

def one_game():
    print("Player 1 trained net: ")
    T = 1
    device = 'cuda'
    network = PolicyValueNet(7,6,2*T)
    network.load_state_dict(torch.load("./realboard.pth",map_location=device))
    network.cuda()
    network.eval()
    winner = network_v_random(network,T,7,6,True)
    print(f"Player {winner} wins!")
watchGame("testGame1")

#one_game()

def eval_random_v_random():
    eval_network(random_v_random,{},1,5,5)

#eval_random_v_random()
    
def eval_random_v_mcts():
    eval_network(random_v_mcts, {}, 1,5,5)

#eval_random_v_mcts()

