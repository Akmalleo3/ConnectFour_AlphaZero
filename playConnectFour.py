import shutil
import os
import time
from numpy import random

from connectFour import ConnectFour

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

#clear output directory if it exists
shutil.rmtree("testGame1", ignore_errors=True)

cf = ConnectFour(5,5,True,"testGame1")

rng = random.default_rng()
player = 1

while(not cf.gameTie()):
    actions = cf.legalMoves()
    #make a random move
    move = rng.integers(len(actions))
    valid = False
    while(not valid):
        valid = cf.move(actions[move])

    winner = cf.gameWinner()
    if winner:
        print(f"Player: {player} WINS!")
        break
    if player == 1:
        player = 2
    else:
        player = 1


watchGame("testGame1")

print(cf.p1_connectedSets)
print(cf.p2_connectedSets)