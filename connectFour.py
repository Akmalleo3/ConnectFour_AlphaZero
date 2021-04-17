
import os
from datetime import datetime
import shutil

from ConnectedSetsManager import ConnectedSetsManager

class ConnectFour():
    def __init__(self, width, height, doSave, gameDir=None, **kwargs):
        self.width = width
        self.height = height
        self.doSave = doSave
        ntiles = width*height
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
        if doSave and not gameDir:
            dt = datetime.now().time()
            self.saveDir = f"./game_{dt}"
            os.mkdir(self.saveDir)
        elif doSave:
            self.saveDir = gameDir
            os.mkdir(self.saveDir)

        #track the turn count
        self.turnNum = 0

    def state(self):
        return self.board

    def player(self):
        return self.turnNum % 2

    def clone(self):
        newGame = ConnectFour(self.width, self.height, False)
        newGame.p1_connectedSets = self.p1_connectedSets
        newGame.p2_connectedSets = self.p2_connectedSets
        newGame.board = self.board
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
        return len(self.legalMoves()) == 0
        
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

