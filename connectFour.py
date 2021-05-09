
import os
from datetime import datetime
import shutil
import torch
from ConnectedSetsManager import ConnectedSetsManager

# Convert a game to an image input for the network
# tensor is of size nxnx(2T) 2 boards for each timestep,
# one for each player's positions at time t, from 0->T end of game
def historyToImage(history,width, height,T):
    cuda = torch.device('cuda')
    state = torch.zeros((2*len(history),width,height),device=cuda)
    t = 0
    for board in history:
        pToVal = {1:1, 2:0, 0:0}
        p1 = torch.tensor([pToVal[val] for val in board.values()])
        p1 = torch.reshape(p1, (width,height))
        pToVal = {1:0, 2:1, 0:0}
        p2 = torch.tensor([pToVal[val] for val in board.values()])
        p2 = torch.reshape(p2, (width,height))
        state[t] = p1
        state[t+1] = p2
        t = t+2

    (n, w,h) = state.shape
    diff = 2*T-n
    if diff > 0:
        state = torch.cat([torch.zeros(diff,w,h,device=cuda),state])
    elif diff < 0:
        state = state[abs(diff):n,:]
    return state.unsqueeze(0)

class ConnectFour():
    def __init__(self, width, height, doSave, doDraw=False, gameDir=None, **kwargs):
        self.width = width
        self.height = height
        self.doSave = doSave
        self.doDraw = doDraw
        ntiles = width*height
        self.history = []
        self.board = {}
        # map col number to height of column 
        self.colPieceCount = {}
        self.p1_connectedSets = []
        self.p2_connectedSets = []
        self.connectedSetManager = ConnectedSetsManager(width,height)

        #All tiles initially unoccupied
        for t in range(ntiles):
            self.board[t] = 0
        for c in range(self.width):
            self.colPieceCount[c] = 0

        # Set up the directory to save each turn
        if doDraw and not gameDir:
            dt = datetime.now().time()
            self.saveDir = f"./game_{dt}"
            os.mkdir(self.saveDir)
        elif doDraw:
            self.saveDir = gameDir
            os.mkdir(self.saveDir)

        #track the turn count
        self.turnNum = 0

    def state(self):
        return self.board

    def player(self): #player 1 or 2
        return self.turnNum % 2 +  1

    def clone(self):
        newGame = ConnectFour(self.width, self.height, False)
        newGame.p1_connectedSets = list(self.p1_connectedSets)
        newGame.p2_connectedSets = list(self.p2_connectedSets)
        newGame.board = dict(self.board)
        newGame.colPieceCount = dict(self.colPieceCount)
        return newGame

    # Drop a piece into specified column
    # for the specified player (1 or 2)
    def move(self,column):
        player = self.player()
        # assert move is legal
        assert(column < self.width)
        assert(player ==1 or player == 2)
        nextRow = self.colPieceCount[column]
        
        if nextRow == self.height:
            return False

        assert(nextRow < self.height)
        
        tileIndex = nextRow*self.width + column 
        
        assert(self.board[tileIndex] == 0)

        self.board[tileIndex] = player
        self.colPieceCount[column]+= 1
        if player == 1:
            self.p1_connectedSets = self.connectedSetManager.insertTile(tileIndex, self.p1_connectedSets)
        else:
            self.p2_connectedSets = self.connectedSetManager.insertTile(tileIndex, self.p2_connectedSets)

        if self.doSave:
            self.history.append(dict(self.board))
        
        if self.doDraw:
            self.draw()

        self.turnNum += 1
        return True 

    #returns 0 if no winner
    #else returns player who won
    def gameWinner(self):
        if self.connectedSetManager.maxConnect(self.p1_connectedSets) >= 4:
            return 1
        if self.connectedSetManager.maxConnect(self.p2_connectedSets) >= 4:
            return 2
        return 0
    
    # return the set of possible next moves
    # corresponds to the columns with an open row to play
    def legalMoves(self):
        availableCols = []
        for (col,row) in self.colPieceCount.items():
            if row < self.height-1:
                availableCols.append(col)
        return availableCols

    #no more moves, game over
    def gameTie(self):
        return (len(self.legalMoves()) == 0)
    
    # for game animation afterwards
    def draw(self):
        if self.turnNum < 10:
            turn = "turn0"
        else:
            turn = "turn"
        turnFile = f"{self.saveDir}/{turn}{self.turnNum}.txt"
        with open(turnFile,'w') as f:
            f.write("Player 1 with X\n")
            f.write("Player 2 with O")
            f.write("\n")
            for i in range(self.height-1,-1,-1):
                for j in range(self.width):
                    loc = i * self.width + j
                    p = self.board[loc]
                    if p == 1:
                        f.write('X'.center(8))
                    elif p == 2:
                        f.write('O'.center(8))
                    else:
                        f.write('_'.center(8))
                f.write('\r\n\r\n')

