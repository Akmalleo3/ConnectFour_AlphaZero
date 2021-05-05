from policyValueNet import PolicyValueNet
import torch 
from connectFour import ConnectFour, historyToImage
from numpy import random as numpy_random
import random 
import os
from mcts import renormalize
import time

cuda = torch.device('cuda')


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


def model_move(model,game, T):
    img = historyToImage(game.history, game.width, game.height,T)

    img = img.unsqueeze(0)
    log_policy,val = model(img)
    policy = torch.exp(log_policy).cpu().detach().numpy()
    moved = False
    while(not moved):
        act = random.choices(list(range(len(policy))),policy)[0]
        moved = game.move(act)
        if not moved:
            policy = renormalize(policy,act)
    return game

def random_move(game):
    rng = numpy_random.default_rng()
    moved = False
    actions = game.legalMoves()
    while(not moved):
        move = rng.integers(len(actions))
        moved = game.move(actions[move])
    return game

def network_v_random(network,T):
    game = ConnectFour(5,5,True)
    while(not game.gameTie()):
        if game.player() == 2:
           game = random_move(game)
        elif game.player() == 1:
            game = model_move(network, game, T)
        if game.gameWinner():
            return game.gameWinner()

    return 0
    '''
    if winner == 1:
        print("Trained Player 1 Wins!")
    elif winner == 2:
        print("Random player triumphed?")
    else:
        print("Game Tied")
    watchGame("Trained_v_Random")
    '''
def random_v_random(nothing, nobody):
    game = ConnectFour(5,5,True)
    while(not game.gameTie()):
        game = random_move(game)
        if game.gameWinner():
            return game.gameWinner()
    return 0


def eval_network(player_scheme, network,T):
    n_trials = 200
    n_wins = 0
    n_draws = 0
    for _ in range(n_trials):
        winner = player_scheme(network,T)
        if winner == 1: 
            n_wins += 1
        elif winner == 0:
            n_draws += 1
    print(f"{n_wins/n_trials} winning rate of Player 1")
    print(f"{n_draws/n_trials} draw rate")


def eval_network_v_random():
    print("Player 1 trained net: ")
    T = 1
    device = 'cuda'
    network = PolicyValueNet(5,5,2*T)
    network.load_state_dict(torch.load("baseline_nonetwork.pth",map_location=device))
    network.cuda()
    network.eval()
    eval_network(network_v_random,network,T)


eval_network_v_random()


def eval_random_v_random():
    eval_network(random_v_random,{},0)

#eval_random_v_random()

def eval_network_v_mcts():
    pass

def eval_mcts_v_mcts():
    pass 

